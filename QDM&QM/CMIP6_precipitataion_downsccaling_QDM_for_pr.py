import numpy as np
import xarray as xr
import time
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d
from scipy.stats import gamma, kstest
import os
import gc
import dask
import pandas as pd
from pathlib import Path

# Configure parallel computing and temporary directory
dask.config.set(scheduler='threads', pool_size=4)
dask.config.set({'temporary-directory': 'G:/dask-temp'})  # Ensure the path exists


# ----------------------------
# Helper function: 1. Safe data loading
# ----------------------------
def load_data_with_checks(
        file_path,
        var_name,
        time_dim='time',
        target_units='mm/day',
        rename_latlon=True
):
    file_path = file_path.replace('\\', '/')
    with xr.set_options(keep_attrs=True, warn_for_unclosed_files=False):
        ds = xr.open_dataset(file_path, chunks={time_dim: 500})

    # Variable existence check
    if var_name not in ds.variables:
        raise KeyError(f"Variable {var_name} not found in {file_path}, existing variables: {list(ds.variables.keys())}")

    # Handle unit suffixes (e.g., mm/day_data → mm/day)
    actual_unit = ds[var_name].attrs.get('units', 'unknown')
    unit_suffixes = ['_data', '_daily', '_dayavg']  # Common suffixes to remove
    for suffix in unit_suffixes:
        if actual_unit.endswith(suffix):
            actual_unit = actual_unit[:-len(suffix)]
            break  # Stop processing after finding matching suffix

    # Unit check (compare processed actual unit with target unit)
    if actual_unit != target_units:
        raise ValueError(
            f"Unit mismatch: requires {target_units}, actual is {ds[var_name].attrs['units']} (processed as {actual_unit})"
        )
    # If unit is correct, update the data's units attribute
    ds[var_name].attrs['units'] = actual_unit

    # Time coordinate conversion
    if not np.issubdtype(ds[time_dim].dtype, np.dtype('datetime64')):
        try:
            ds[time_dim] = pd.to_datetime(ds[time_dim].values)
        except Exception as e:
            raise ValueError(f"Unable to convert {time_dim} to datetime: {str(e)}")

    # Dimension renaming
    if rename_latlon and 'y' in ds.dims and 'lat' not in ds.dims:
        ds = ds.rename({'y': 'lat', 'x': 'lon'})
        print(f"Renamed y/x dimensions to lat/lon for {file_path}")

    return ds[var_name]


# ----------------------------
# Helper function: 2. Model resampling to observation grid
# ----------------------------
def resample_model_to_obs_grid(model_data, obs_lat, obs_lon, var_name='pr'):
    """Interpolate and resample model data to observation data's lat/lon grid"""
    print(f"\nStarting model resampling: target grid (lat: {len(obs_lat)} points, lon: {len(obs_lon)} points)")
    print(f"Original model grid: lat {model_data['lat'].min().values:.4f}~{model_data['lat'].max().values:.4f}")
    print(f"Target observation grid: lat {obs_lat.min().values:.4f}~{obs_lat.max().values:.4f}")

    resampled = model_data.interp(
        lat=obs_lat,
        lon=obs_lon,
        method='linear',
        kwargs={'fill_value': np.nan}
    )

    resampled.attrs = model_data.attrs
    resampled.name = var_name

    print("Model resampling completed!")
    return resampled


def enforce_strict_increasing(q_array, min_diff=1e-8):
    """Ensure quantiles are strictly increasing"""
    q = q_array.copy()
    for i in range(1, len(q)):
        if q[i] <= q[i - 1]:
            q[i] = q[i - 1] + min_diff
    return q


# ----------------------------
# QDM correction core algorithm
# ----------------------------
def non_parametric_BC_QDM_rolling_parallel(obs_1d, model_hist_1d, model_fut_1d):
    obs_flat = obs_1d.ravel().astype(np.float32)
    model_hist_flat = model_hist_1d.ravel().astype(np.float32)
    model_fut_flat = model_fut_1d.ravel().astype(np.float32)
    original_nan_mask = np.isnan(model_fut_flat)  # Preserve original nan mask

    nbins = 1000
    bin_edges = np.linspace(0, 1, nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Observation quantile calculation
    mask_obs = obs_flat >= 0.05
    valid_obs = obs_flat[mask_obs]
    if len(valid_obs) < 100:
        # When data is insufficient, return clipped original values but preserve nan
        result = np.clip(model_fut_1d, 0, 1e4).astype(np.float32)
        return np.where(original_nan_mask, np.nan, result)

    # Gamma fitting and KS test
    params = gamma.fit(valid_obs, floc=0)
    stat, p_value = kstest(valid_obs, 'gamma', args=params)
    gamma_fit_ok = p_value > 0.05

    # Mixed quantiles
    high_quantile_thres = 0.95
    high_bins = bin_centers >= high_quantile_thres
    q_obs_emp = mquantiles(valid_obs, bin_centers[~high_bins])
    if gamma_fit_ok:
        q_obs_gamma = gamma.ppf(bin_centers[high_bins], *params)
        q_obs = np.concatenate([q_obs_emp, q_obs_gamma])
    else:
        q_obs = mquantiles(valid_obs, bin_centers)
    q_obs = enforce_strict_increasing(q_obs)

    # Historical model quantiles
    mask_hist = model_hist_flat >= 0.05
    valid_hist = model_hist_flat[mask_hist]
    if len(valid_hist) < 100:
        result = np.clip(model_fut_1d, 0, 1e4).astype(np.float32)
        return np.where(original_nan_mask, np.nan, result)
    q_hist = mquantiles(valid_hist, bin_centers)
    q_hist = enforce_strict_increasing(q_hist)

    # Future model quantiles
    mask_fut = model_fut_flat >= 0.05
    valid_fut = model_fut_flat[mask_fut]
    if len(valid_fut) < 100:
        result = np.clip(model_fut_1d, 0, 1e4).astype(np.float32)
        return np.where(original_nan_mask, np.nan, result)
    q_fut = mquantiles(valid_fut, bin_centers)
    q_fut = enforce_strict_increasing(q_fut)

    # Future quantile slope (for extrapolation)
    delta_q_fut = max(
        (q_fut[-1] - q_fut[0]) / (bin_centers[-1] - bin_centers[0]),
        1e-8
    )

    # ----------------------------
    # 1. First use nearest neighbor to fill boundaries, then manually calculate extrapolated values
    # ----------------------------
    # Future → probability (use edge values to fill boundaries, avoid using function as parameter)
    fut_to_p = interp1d(
        q_fut, bin_centers,
        bounds_error=False,
        fill_value=(bin_centers[0], bin_centers[-1]),  # Fill with first and last values
        assume_sorted=True
    )
    p_values = fut_to_p(model_fut_flat)

    # 2. Manually calculate extrapolated values
    # Extrapolation below lower bound
    below_mask = model_fut_flat < q_fut[0]
    p_values[below_mask] = np.maximum(
        0.0,
        bin_centers[0] + (model_fut_flat[below_mask] - q_fut[0]) / delta_q_fut
    )
    # Extrapolation above upper bound
    above_mask = model_fut_flat > q_fut[-1]
    p_values[above_mask] = np.minimum(
        1.0,
        bin_centers[-1] + (model_fut_flat[above_mask] - q_fut[-1]) / delta_q_fut
    )
    p_values = np.clip(p_values, 0.0, 1.0)  # Ensure within [0,1] range

    # Probability → observation/historical values
    hist_from_p = interp1d(
        bin_centers, q_hist,
        bounds_error=False,
        fill_value=(q_hist[0], q_hist[-1])
    )
    obs_from_p = interp1d(
        bin_centers, q_obs,
        bounds_error=False,
        fill_value=(q_obs[0], q_obs[-1])
    )

    # Delta factor calculation
    hist_val = np.maximum(hist_from_p(p_values), 1e-3)
    obs_val = obs_from_p(p_values)
    delta = np.zeros_like(model_fut_flat)
    valid_delta_mask = (model_fut_flat > 1e-6) & (hist_val > 1e-6)
    delta[valid_delta_mask] = np.clip(
        model_fut_flat[valid_delta_mask] / hist_val[valid_delta_mask],
        0, 100
    )

    # Correction and wet/dry day matching
    corrected = obs_val * delta
    p_obs_wet = np.mean(mask_obs)
    p_model_wet = np.mean(mask_fut)
    if p_model_wet > 1e-6 and p_obs_wet < p_model_wet:
        adjust_quantile = 1 - (p_obs_wet / p_model_wet)
        adjust_threshold = np.quantile(corrected[~original_nan_mask], adjust_quantile)  # Calculate threshold ignoring nan
        corrected = np.where(corrected < adjust_threshold, 0.0, corrected)

    # Post-processing: preserve original nan regions
    corrected = np.where(
        original_nan_mask,
        np.nan,
        np.where(
            model_fut_flat > 0.05,
            np.clip(corrected, 0, 1e4),
            model_fut_1d  # Dry days preserve original value (0)
        )
    )
    return corrected.reshape(model_fut_1d.shape).astype(np.float32)


# ----------------------------
# Time series correction main program
# ----------------------------
def correct_full_timeseries_parallel(
        obs_3d,
        model_hist_3d,
        model_fut_3d,
        output_dir,
        model_var_name='pr'
):
    # Initialize correction result array, preserving original nan structure
    final_corrected = model_fut_3d.copy(deep=True)
    max_year = 2100
    os.makedirs(output_dir, exist_ok=True)

    # Resume from breakpoint
    processed_years = set()
    for fname in os.listdir(output_dir):
        if fname.startswith("corrected_") and fname.endswith(".nc"):
            try:
                year = int(fname.split("_")[1].split(".")[0])
                file_path = os.path.join(output_dir, fname)
                if os.path.getsize(file_path) > 1024:
                    processed_years.add(year)
            except (ValueError, IndexError):
                continue
    print(f"Processed years: {sorted(processed_years) if processed_years else 'None'}")

    # Yearly loop
    for target_year in range(2015, max_year + 1):
        if target_year in processed_years:
            print(f"\nSkipping already processed year: {target_year}")
            continue

        year_start = time.time()
        print(f"\n{'=' * 50}\nStarting processing year: {target_year}\n{'=' * 50}")

        # Sliding window length strictly 30 years
        window_size = 30
        half_window = (window_size - 1) // 2  # 14, ensuring window length = 30 years

        # Get actual time range of model future period
        data_min_year = model_fut_3d['time2'].dt.year.min().item()
        data_max_year = model_fut_3d['time2'].dt.year.max().item()

        # Calculate ideal window
        ideal_start = target_year - half_window
        ideal_end = target_year + (window_size - 1 - half_window)

        # Adjust window to be within data range
        if ideal_start < data_min_year:
            window_start = data_min_year
            window_end = data_min_year + window_size - 1
        elif ideal_end > data_max_year:
            window_end = data_max_year
            window_start = data_max_year - window_size + 1
        else:
            window_start = ideal_start
            window_end = ideal_end

        # Validate window length
        if (window_end - window_start + 1) != window_size:
            raise ValueError(
                f"Window length error: actual {window_end - window_start + 1} years, expected {window_size} years, "
                f"please check model data time range or window calculation logic"
            )

        print(f"Current sliding window: {window_start}-{window_end} ({window_end - window_start + 1} years), target year: {target_year}")

        # Extract future model data within sliding window
        fut_roll = model_fut_3d.sel(
            time2=slice(f"{window_start}-01-01", f"{window_end}-12-31")
        )

        # Verify sliding window contains target year
        if not (window_start <= target_year <= window_end):
            raise ValueError(
                f"Sliding window error: {window_start}-{window_end} does not contain target year {target_year}, "
                f"please check if model data time range is correct"
            )

        # Process by month
        for month in range(1, 13):
            month_start = time.time()
            output_path = os.path.join(output_dir, f"corrected_{target_year}.nc")

            # Skip already processed months
            if os.path.exists(output_path):
                with xr.open_dataset(output_path) as ds:
                    if len(ds['time2'].dt.month.sel(time2=f"{target_year}-{month:02d}")) > 0:
                        print(f"Skipping already processed month: {target_year}-{month:02d}")
                        continue

            print(f"\nProcessing {target_year}-{month:02d}...")

            # Extract monthly data
            obs_month = obs_3d.where(obs_3d['time'].dt.month == month, drop=True)
            hist_month = model_hist_3d.where(model_hist_3d['time1'].dt.month == month, drop=True)
            fut_month = fut_roll.where(fut_roll['time2'].dt.month == month, drop=True)

            # Check if monthly data exists
            if len(fut_month['time2']) == 0:
                print(f"Warning: No {month} month data in sliding window {window_start}-{window_end}, skipping")
                continue

            # Adjust chunking
            obs_month = obs_month.chunk({'time': -1, 'lat': 10, 'lon': 10})
            hist_month = hist_month.chunk({'time1': -1, 'lat': 10, 'lon': 10})
            fut_month = fut_month.chunk({'time2': -1, 'lat': 10, 'lon': 10})

            # Parallel correction
            corrected = xr.apply_ufunc(
                non_parametric_BC_QDM_rolling_parallel,
                obs_month,
                hist_month,
                fut_month,
                input_core_dims=[['time'], ['time1'], ['time2']],
                output_core_dims=[['time2']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[np.float32]
            )

            # Update results
            time_mask = (
                    (final_corrected['time2'].dt.year == target_year) &
                    (final_corrected['time2'].dt.month == month)
            )
            target_time = final_corrected['time2'].where(time_mask, drop=True)
            if len(target_time) == 0:
                print(f"Warning: No data for {target_year}-{month:02d}, skipping")
                continue

            # Verify target time is within sliding window
            target_year_in_window = (target_time.dt.year >= window_start) & (target_time.dt.year <= window_end)
            if not np.all(target_year_in_window):
                invalid_times = target_time[~target_year_in_window]
                print(f"Warning: The following times are not in the sliding window and will be skipped: {invalid_times.dt.strftime('%Y-%m-%d').values}")
                target_time = target_time[target_year_in_window]
                if len(target_time) == 0:
                    print(f"Warning: No valid data in sliding window for {target_year}-{month:02d}, skipping")
                    continue

            # Extract and update correction results
            try:
                corrected_target = corrected.sel(time2=target_time).compute()
            except KeyError as e:
                print(f"Time matching error: {str(e)}")
                print(
                    f"Sliding window time range: {fut_roll['time2'].min().dt.strftime('%Y-%m-%d').values} ~ {fut_roll['time2'].max().dt.strftime('%Y-%m-%d').values}")
                print(f"Target time: {target_time.dt.strftime('%Y-%m-%d').values}")
                continue

            update_slice = corrected_target.transpose('time2', 'lat', 'lon')
            if update_slice.shape != final_corrected.loc[dict(time2=update_slice.time2)].shape:
                raise ValueError(
                    f"Dimension mismatch: correction result {update_slice.shape} vs target position {final_corrected.loc[dict(time2=update_slice.time2)].shape}"
                )
            final_corrected.loc[dict(time2=update_slice.time2)] = update_slice.values

            # Memory cleanup
            del corrected, corrected_target, obs_month, hist_month, fut_month
            gc.collect()

            # Progress report
            month_elapsed = time.time() - month_start
            print(f"{target_year}-{month:02d} processing completed, time taken: {month_elapsed:.1f} seconds")

        # Save yearly data
        if not os.path.exists(output_path):
            yearly_data = final_corrected.sel(
                time2=slice(f"{target_year}-01-01", f"{target_year}-12-31")
            )
            tmp_path = output_path.replace(".nc", ".tmp")
            yearly_data.to_netcdf(
                tmp_path,
                mode='w',
                encoding={model_var_name: {'zlib': True, 'complevel': 5}},
                format='NETCDF4'
            )
            try:
                os.rename(tmp_path, output_path)
                print(f"\n{target_year} data saved to: {output_path}")
            except PermissionError:
                print(f"\nWarning: {output_path} already exists, cannot overwrite")

        # Yearly time consumption
        year_elapsed = (time.time() - year_start) / 60
        print(f"Total time for {target_year}: {year_elapsed:.1f} minutes")

    # Final post-processing: first preserve original invalid region nans, then process dry days and range
    final_corrected = final_corrected.where(~model_fut_3d.isnull())  # Priority to preserve original nan
    final_corrected = xr.where(
        (final_corrected > 0.05) | final_corrected.isnull(),  # Only process dry days in valid data regions
        final_corrected,
        0.0
    ).clip(max=1e4)  # Precipitation value upper limit clipping

    return final_corrected


# ----------------------------
# Main program entry
# ----------------------------
if __name__ == "__main__":
    # 1. Configuration parameters
    OBS_PATH = r'E:\Biyelunwen\obs_data\CMFD_v2.0_obs_cut_verify\variable_data_1961-2014\prec\masked_prec_CMFD_NPM_V0200_B-01_01dy_010deg_19610101-20141231_converted.nc'
    MODEL_ROOT = r"G:\ghq_resample_MME_qdm\NPM\pr\ssp245"
    OUTPUT_BASE = r"G:\ghq_resample_qdm_downscaled\NPM\pr\V2\ssp245"
    OBS_VAR = 'prec'
    MODEL_VAR = 'pr'
    TARGET_UNITS = 'mm/day'
    OBS_TIME_SLICE = slice('1961-01-01', '2014-12-31')
    MODEL_HIST_SLICE = slice('1961-01-01', '2014-12-31')
    MODEL_FUT_SLICE = slice('2015-01-01', '2100-12-31')
    RESAMPLE_TOLERANCE = 1e-3

    # 2. Load observation data
    print("Loading observation data...")
    obs_data = load_data_with_checks(
        file_path=OBS_PATH,
        var_name=OBS_VAR,
        target_units=TARGET_UNITS,
        rename_latlon=True
    ).sel(time=OBS_TIME_SLICE)

    # Observation data preprocessing: set dry days to 0, preserve original nan
    obs_data = obs_data.where(obs_data >= 0.05, 0.0)
    print(f"Observation data information:")
    print(f"  Variable name: {OBS_VAR}")
    print(f"  Dimensions: {obs_data.dims}")
    print(
        f"  Time range: {obs_data['time'].min().dt.strftime('%Y-%m-%d').values} ~ {obs_data['time'].max().dt.strftime('%Y-%m-%d').values}")
    print(
        f"  Spatial range: lat {obs_data['lat'].min().values:.4f}~{obs_data['lat'].max().values:.4f} ({len(obs_data['lat'])} points)")
    print(
        f"  Spatial range: lon {obs_data['lon'].min().values:.4f}~{obs_data['lon'].max().values:.4f} ({len(obs_data['lon'])} points)")

    # 3. Traverse and load model data with automatic resampling
    model_files = list(Path(MODEL_ROOT).glob("*.nc")) + list(Path(MODEL_ROOT).glob("*.nc4"))
    if not model_files:
        raise FileNotFoundError(f"No model files (.nc/.nc4) found in {MODEL_ROOT}")

    for model_path in model_files:
        model_name = model_path.name
        print(f"\n{'#' * 60}\nStarting processing model: {model_name}\n{'#' * 60}")

        # Extract model scenario
        if 'ssp585' in model_name.lower():
            scenario = 'ssp585'
        elif 'ssp126' in model_name.lower():
            scenario = 'ssp126'
        elif 'historical' in model_name.lower():
            scenario = 'historical'
        else:
            scenario = 'unknown'
        print(f"Model scenario: {scenario}")

        # Load model raw data
        print("Loading model historical period data...")
        model_hist_raw = load_data_with_checks(
            file_path=str(model_path),
            var_name=MODEL_VAR,
            target_units=TARGET_UNITS,
            rename_latlon=True
        ).sel(time=MODEL_HIST_SLICE)

        print("Loading model future period data...")
        model_fut_raw = load_data_with_checks(
            file_path=str(model_path),
            var_name=MODEL_VAR,
            target_units=TARGET_UNITS,
            rename_latlon=True
        ).sel(time=MODEL_FUT_SLICE)

        # Model resampling to observation grid
        print("\n" + "=" * 30 + " Model Resampling " + "=" * 30)
        model_hist_resampled = resample_model_to_obs_grid(
            model_data=model_hist_raw,
            obs_lat=obs_data['lat'],
            obs_lon=obs_data['lon'],
            var_name=MODEL_VAR
        ).rename({'time': 'time1'})

        model_fut_resampled = resample_model_to_obs_grid(
            model_data=model_fut_raw,
            obs_lat=obs_data['lat'],
            obs_lon=obs_data['lon'],
            var_name=MODEL_VAR
        ).rename({'time': 'time2'})

        # Model data preprocessing: set dry days to 0, while preserving original nan
        model_hist = model_hist_resampled.where(model_hist_resampled >= 0.05, 0.0)
        model_hist = model_hist.where(model_hist_resampled.notnull())  # Restore original invalid region nan

        model_fut = model_fut_resampled.where(model_fut_resampled >= 0.05, 0.0)
        model_fut = model_fut.where(model_fut_resampled.notnull())  # Restore original invalid region nan

        # Validate spatial consistency after resampling
        lat_match = np.allclose(model_hist['lat'].values, obs_data['lat'].values, rtol=RESAMPLE_TOLERANCE)
        lon_match = np.allclose(model_hist['lon'].values, obs_data['lon'].values, rtol=RESAMPLE_TOLERANCE)
        if not (lat_match and lon_match):
            raise ValueError(f"Spatial grid still mismatched after resampling! lat match: {lat_match}, lon match: {lon_match}")
        print("✅ Spatial grid perfectly matches observation after resampling")

        # Print model data information
        print(f"\nModel historical period (after resampling) information:")
        print(f"  Dimensions: {model_hist.dims}")
        print(
            f"  Time range: {model_hist['time1'].min().dt.strftime('%Y-%m-%d').values} ~ {model_hist['time1'].max().dt.strftime('%Y-%m-%d').values}")
        print(f"Model future period (after resampling) information:")
        print(f"  Dimensions: {model_fut.dims}")
        print(
            f"  Time range: {model_fut['time2'].min().dt.strftime('%Y-%m-%d').values} ~ {model_fut['time2'].max().dt.strftime('%Y-%m-%d').values}")

        # 4. Execute QDM correction
        func_name = non_parametric_BC_QDM_rolling_parallel.__name__
        output_dir = os.path.join(OUTPUT_BASE, f"{scenario}_{func_name}")
        print(f"\nCorrection result output directory: {output_dir}")

        print("Starting QDM bias correction...")
        corrected_result = correct_full_timeseries_parallel(
            obs_3d=obs_data,
            model_hist_3d=model_hist,
            model_fut_3d=model_fut,
            output_dir=output_dir,
            model_var_name=MODEL_VAR
        )

        # 5. Save final results
        final_output_path = os.path.join(output_dir, f"corrected_{scenario}_full.nc")
        corrected_result.to_netcdf(
            final_output_path,
            mode='w',
            encoding={MODEL_VAR: {'zlib': True, 'complevel': 5}},
            format='NETCDF4'
        )
        print(f"\nComplete correction result saved to: {final_output_path}")
        print(f"{'#' * 60}\nModel {model_name} processing completed\n{'#' * 60}")

        # Memory cleanup
        del model_hist_raw, model_fut_raw, model_hist, model_fut, corrected_result
        gc.collect()

    print("\nAll model processing completed!")
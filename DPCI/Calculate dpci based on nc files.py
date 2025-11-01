import numpy as np
import xarray as xr
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import os  # Added: For file path handling


def calculate_ci(annual_precip):
    """Calculate annual CI (Climate Imprint) value (logic remains unchanged)"""
    precip_flat = annual_precip.flatten()
    precip_flat = precip_flat[~np.isnan(precip_flat)]
    precip_flat = precip_flat[precip_flat > 0.1]  # Filter negligible precipitation
    if len(precip_flat) == 0:
        return np.nan

    max_value = precip_flat.max()
    classes = np.arange(0, max_value, 1)  # 1mm interval classification
    results = []

    for i in range(len(classes) - 1):
        lower, upper = classes[i], classes[i + 1]
        mask = (precip_flat > lower) & (precip_flat <= upper)
        PM = (lower + upper) / 2  # Midpoint of the precipitation interval
        ni = mask.sum()  # Number of days in the interval
        Pi = PM * ni  # Total precipitation in the interval
        if ni > 0:
            results.append([PM, ni, Pi])

    if not results:
        return np.nan

    results_df = pd.DataFrame(results, columns=['PM', 'ni', 'Pi'])
    results_df['Cumulative_ni'] = results_df['ni'].cumsum()  # Cumulative rainy days
    results_df['Cumulative_Pi'] = results_df['Pi'].cumsum()  # Cumulative precipitation

    total_ni = results_df['ni'].sum()
    total_Pi = results_df['Pi'].sum()
    results_df['ni%'] = results_df['Cumulative_ni'] / total_ni * 100  # Cumulative days percentage
    results_df['Pi%'] = results_df['Cumulative_Pi'] / total_Pi * 100  # Cumulative precipitation percentage

    # Exponential fitting function
    def exp_function(X, b, c):
        return X * np.exp(-b * (100 - X) ** c)

    x_data = results_df['ni%']
    y_data = results_df['Pi%']
    try:
        popt, _ = curve_fit(exp_function, x_data, y_data, bounds=(0, [np.inf, 2]))
    except:
        return np.nan  # Return NaN if fitting fails
    b, c = popt

    # Calculate area for CI computation
    x_new = np.linspace(0, 100, 500)
    y_fit = exp_function(x_new, b, c)
    area_under_curve = simpson(y_fit, x_new)
    area_total = 5000  # Area under equidistribution line (100x100/2)
    CI = (area_total - area_under_curve) / area_total  # Climate Imprint
    return CI


def process_single_nc(input_file, output_file):
    """Process a single NetCDF file and generate CI (Climate Imprint) results"""
    data = xr.open_dataset(input_file)
    precip = data['pr']  # Precipitation variable (assumed unit: mm/day)

    # Retrieve time information from the file (more general than manual generation)
    time = pd.to_datetime(data['time'].values)
    years = np.unique(time.year)

    ci_list = []
    for year in years:
        mask = time.year == year
        annual_precip = precip.sel(time=mask).values
        # Apply CI calculation to each grid point (lat/lon)
        ci_map = np.apply_along_axis(calculate_ci, axis=0, arr=annual_precip)

        # Validate CI range (optional)
        if np.any((ci_map < 0) & ~np.isnan(ci_map)):
            print(f"Warning: CI values out of valid range [0,1] for year {year} in {input_file}")

        # Convert to xarray DataArray with coordinates
        ci_map = xr.DataArray(
            ci_map,
            name='CI',
            dims=['lat', 'lon'],
            coords={'lat': precip['lat'], 'lon': precip['lon']}
        )
        ci_list.append((year, ci_map))

    # Sort by year and concatenate results
    sorted_ci_list = sorted(ci_list, key=lambda x: x[0])
    years_sorted = [year for year, _ in sorted_ci_list]
    ci_maps_sorted = [ci_map for _, ci_map in sorted_ci_list]
    CI_combined = xr.concat(ci_maps_sorted, pd.Index(years_sorted, name='year'))

    # Save output NetCDF file
    CI_combined.to_netcdf(output_file)
    print(f"Processed {input_file} -> {output_file}")


def batch_process_nc_files(input_dir, output_dir):
    """Batch process all .nc files in the input directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all .nc files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.nc'):
            input_file = os.path.join(input_dir, filename)
            # Generate output filename (e.g., "original.nc" -> "original_ci.nc")
            output_filename = f"{filename.split('.nc')[0]}_ci.nc"
            output_file = os.path.join(output_dir, output_filename)
            process_single_nc(input_file, output_file)


# Configuration (User-specified paths)
INPUT_DIR = r' '
OUTPUT_DIR = r' '

# Execute batch processing
batch_process_nc_files(INPUT_DIR, OUTPUT_DIR)
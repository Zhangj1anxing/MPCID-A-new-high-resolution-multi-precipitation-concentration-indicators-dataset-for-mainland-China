import xarray as xr
import numpy as np
from math import atan2, degrees


def calculate_pcp_netcdf(input_path, output_path):
    """
    Process NetCDF files to calculate the Precipitation Concentration Period (PCP)
    Input requirements:
    - Contains time dimension (time) and precipitation variable (pr)
    - Time coordinates must be parsable as datetime
    """
    # Define monthly angle mapping (Table II in the paper)
    month_angles = {
        1: 0, 2: 30, 3: 60,
        4: 90, 5: 120, 6: 150,
        7: 180, 8: 210, 9: 240,
        10: 270, 11: 300, 12: 330
    }

    # Open dataset and process time
    ds = xr.open_dataset(input_path)

    # Ensure time coordinates are correctly parsed
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        ds = xr.decode_cf(ds)

    # Extract year and month information
    years = ds.time.dt.year.rename('year')
    months = ds.time.dt.month

    # Calculate radians for corresponding angles for each time step
    angles_deg = xr.DataArray(
        [month_angles[m.item()] for m in months],
        dims=['time'],
        coords={'time': ds.time}
    )
    angles_rad = np.deg2rad(angles_deg)

    # Calculate vector components for each time step
    pr_sin = ds.pr * np.sin(angles_rad)
    pr_cos = ds.pr * np.cos(angles_rad)

    # Aggregate by year (retain all grid points)
    sum_sin = pr_sin.groupby(years).sum(dim='time', skipna=True)
    sum_cos = pr_cos.groupby(years).sum(dim='time', skipna=True)
    total_precip = ds.pr.groupby(years).sum(dim='time', skipna=True)

    # Calculate resultant vector angle
    pcp_rad = xr.apply_ufunc(atan2, sum_sin, sum_cos,
                             input_core_dims=[[], []],
                             output_core_dims=[[]],
                             vectorize=True)
    pcp_deg = np.rad2deg(pcp_rad)

    # Adjust angle to 0-360 range
    pcp_deg = pcp_deg.where(pcp_deg >= 0, pcp_deg + 360)

    # Handle cases with no precipitation
    pcp_deg = pcp_deg.where(total_precip > 0)

    # Map to specific months
    pcp_month = xr.full_like(pcp_deg, np.nan, dtype=float)

    for month in month_angles:
        ma = month_angles[month]
        lower = (ma - 15) % 360
        upper = (ma + 15) % 360

        if lower < upper:
            cond = (pcp_deg >= lower) & (pcp_deg < upper)
        else:
            cond = (pcp_deg >= lower) | (pcp_deg < upper)

        pcp_month = xr.where(cond, month, pcp_month)

    # Create output dataset
    ds_out = xr.Dataset({
        'pcp_angle': pcp_deg.round(2),  # Keep 2 decimal places
        'pcp_month': pcp_month.astype('int16')  # Save storage space
    })

    # Add coordinate descriptions
    ds_out['year'] = sum_sin.year
    ds_out['lat'] = ds.lat
    ds_out['lon'] = ds.lon

    # Set encoding (optional)
    encoding = {
        'pcp_angle': {'dtype': 'float32', 'zlib': True},
        'pcp_month': {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
    }

    # Save results
    ds_out.to_netcdf(output_path, encoding=encoding)
    print(f"Processing completed, results saved to {output_path}")


# Example usage
input_path = "E:\\RE_QDM&QM_downscaling\\qdm_resample025_pr_day_MME_fut_ssp126_r1i1p1f1_gn_china_region_20150101-21001231.nc"
output_path = "E:\\RE_QDM&QM_downscaling\\PCP\\monthly\\PCP_qdm_ssp126_result.nc"
calculate_pcp_netcdf(input_path, output_path)
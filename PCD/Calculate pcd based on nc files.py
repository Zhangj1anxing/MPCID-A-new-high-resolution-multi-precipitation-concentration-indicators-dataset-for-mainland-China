import numpy as np
import xarray as xr
import pandas as pd


def calculate_pcd(precip, time):
    # Define time angles by month
    months = time.month
    theta = 2 * np.pi * (months - 1) / 12

    # Convert to numpy array and add new axes to match the time dimension shape of precip
    theta = theta.values[:, np.newaxis, np.newaxis]

    # Calculate the sum of x and y components
    R_cos_theta = (precip * np.cos(theta)).sum(dim='time')
    R_sin_theta = (precip * np.sin(theta)).sum(dim='time')

    # Calculate total annual precipitation
    P_total = precip.sum(dim='time')

    # Calculate Precipitation Concentration Degree (PCD)
    PCD = (1 / P_total) * np.sqrt(R_cos_theta ** 2 + R_sin_theta ** 2)

    return PCD


def process_nc_file(input_path, output_path):
    # Read NetCDF data
    data = xr.open_dataset(input_path)
    precip = data['pr']

    # Ensure the time dimension exists and convert to datetime format
    time = pd.to_datetime(data['time'].values)

    # Extract unique years for grouping
    years = np.unique(time.year)

    # Initialize list to store PCD results
    pcd_list = []

    for year in years:
        # Filter data for the specific year
        mask = time.year == year
        annual_precip = precip.sel(time=mask)
        annual_time = time[mask]

        # Calculate PCD for the year
        PCD = calculate_pcd(annual_precip, annual_time)

        # Validate value range (PCD should be between 0 and 1)
        assert PCD.min() >= 0 and PCD.max() <= 1, f"PCD values are out of expected range [0, 1] for year {year}"

        # Add year dimension if missing
        if len(PCD.shape) == 2:
            PCD = PCD.expand_dims(dim='year')

        # Create DataArray with specified name and coordinates
        PCD = xr.DataArray(
            PCD,
            name='PCD',
            dims=['year', 'lat', 'lon'],
            coords={'year': [year], 'lat': annual_precip['lat'], 'lon': annual_precip['lon']}
        )

        pcd_list.append(PCD)

        # Print progress message
        print(f"PCD calculation for year {year} is complete.")

    # Concatenate results and sort by year
    PCD_combined = xr.concat(pcd_list, dim='year').sortby('year')

    # Save to NetCDF file
    PCD_combined.to_netcdf(output_path)


# Set input and output file paths
input_path = 'F:\ssp585resampled025(2015-2100)\MMEs\\qdm_resample025_pr_day_MME_fut_ssp585_r1i1p1f1_gn_china_region_20150101-21001231.nc'  # Update input file path
output_path = 'F:\ssp585resampled025(2015-2100)\MMEs\\PCD_qdm_resample025_pr_day_MME_fut_ssp585_r1i1p1f1_gn_china_region_20150101-21001231.nc'  # Update output folder and filename

# Process the NetCDF file
process_nc_file(input_path, output_path)

print(f"All PCD calculations are complete and saved to {output_path}.")
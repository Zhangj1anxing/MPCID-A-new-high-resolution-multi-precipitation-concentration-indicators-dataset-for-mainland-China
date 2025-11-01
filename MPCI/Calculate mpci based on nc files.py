import numpy as np
import xarray as xr
import pandas as pd
import os


def process_pci_file(input_path, output_path):
    """
    Process NetCDF files to calculate the Precipitation Concentration Index (PCI)
    PCI is computed based on the distribution of monthly precipitation within each year
    """
    # Read NetCDF data
    data = xr.open_dataset(input_path)
    precip = data['pr']  # Precipitation variable (assumed unit: mm/day)

    # Ensure time dimension exists and convert to datetime format
    time = pd.to_datetime(data['time'].values)

    # Extract unique years for annual grouping
    years = np.unique(time.year)

    # Initialize lists to store PCI results and corresponding years
    pci_list = []
    years_list = []

    for year in years:
        # Filter data for the specific year
        mask = time.year == year
        annual_precip = precip.sel(time=mask)

        # Aggregate precipitation by month to get total monthly precipitation
        monthly_precip = annual_precip.groupby('time.month').sum(dim='time')

        # Calculate sum of squared monthly precipitation and squared total annual precipitation
        monthly_precip_squared = (monthly_precip ** 2).sum(dim='month')  # Sum of squared monthly precipitation
        P_total_squared = (monthly_precip.sum(dim='month')) ** 2         # Squared total annual precipitation

        # Calculate Precipitation Concentration Index (PCI)
        PCI = (monthly_precip_squared / P_total_squared) * 100

        # Validate reasonable range of PCI values (PCI ≥ 0)
        assert PCI.min() >= 0, f"PCI values are out of expected range [0, ∞] for year {year}"

        # Create DataArray for PCI results with specified name and coordinates
        PCI = xr.DataArray(
            PCI,
            name='PCI',
            dims=['lat', 'lon'],
            coords={'lat': precip['lat'], 'lon': precip['lon']}
        )

        # Append results to lists
        pci_list.append(PCI)
        years_list.append(year)

    # Sort results by year
    sorted_indices = np.argsort(years_list)
    sorted_pci_list = [pci_list[i] for i in sorted_indices]
    sorted_years_list = [years_list[i] for i in sorted_indices]

    # Concatenate PCI results across years and add year dimension
    PCI_combined = xr.concat(sorted_pci_list, pd.Index(sorted_years_list, name='year'))

    # Save combined results to NetCDF file
    PCI_combined.to_netcdf(output_path)


# Set input and output file paths
input_path = ' '  # Update input file path
output_path = ' '  # Update output folder and filename

# Process the NetCDF file
process_pci_file(input_path, output_path)

print(f"Saved combined PCI results to {output_path}")
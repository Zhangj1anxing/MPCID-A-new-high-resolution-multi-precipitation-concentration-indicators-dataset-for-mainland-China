import os
import pandas as pd
import numpy as np

def calculate_pcd(data):
    # Convert month to azimuth in radians
    data['azimuth'] = 2 * np.pi * data['month'] / 12

    # Calculate the x and y components
    data['x_component'] = data['precipitation'] * np.sin(data['azimuth'])
    data['y_component'] = data['precipitation'] * np.cos(data['azimuth'])

    # Aggregate x and y components by year
    rx = data.groupby('year')['x_component'].sum()
    ry = data.groupby('year')['y_component'].sum()

    # Calculate total annual precipitation
    total_precipitation = data.groupby('year')['precipitation'].sum()

    # Calculate Precipitation Concentration Degree (PCD)
    pcd = np.sqrt(rx**2 + ry**2) / total_precipitation

    # Create result DataFrame
    result = pd.DataFrame({'year': total_precipitation.index, 'pcd': pcd}).reset_index(drop=True)

    return result

def process_files(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read data
            data = pd.read_csv(input_path)

            # Check for required columns
            if not {'year', 'month', 'precipitation'}.issubset(data.columns):
                raise ValueError(f"Input file {filename} is missing required columns: 'year', 'month', 'precipitation'")

            # Calculate PCD
            result = calculate_pcd(data)

            # Save results
            result.to_csv(output_path, index=False)

# Set input and output folder paths
input_folder = r' '  # Note: Update path to your actual input directory
output_folder = r' '  # Note: Update path to your actual output directory

# Process all CSV files in the folder
process_files(input_folder, output_folder)
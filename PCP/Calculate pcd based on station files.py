import os
import pandas as pd
import numpy as np
from math import radians, degrees


def calculate_pcp_station_monthly(input_dir, output_dir):
    """
    Calculate the Precipitation Concentration Period (PCP) based on monthly precipitation data
    :param input_dir: Path to the input directory (containing CSV files starting with "PRE")
    :param output_dir: Path to the output directory for results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all station files in the input directory
    for filename in os.listdir(input_dir):
        if not filename.startswith("PRE") or not filename.endswith(".csv"):
            continue

        # Read station data
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)

        # Preprocessing: check for necessary columns
        if not {'year', 'month'}.issubset(df.columns):
            print(f"Skipping file {filename}: missing required columns")
            continue

        # Remove NaN data
        df = df.dropna(subset=['precipitation'])

        # Process station data
        results = []

        # Calculate by year group
        for year, group in df.groupby('year'):
            try:
                year = int(year)
                if year < 1961 or year > 2020:
                    continue

                # Assign fixed angles to each month, starting from 345° for January
                month_angles = {i: radians((i - 1) * 30 + 345) if i - 1 < 12 else radians((i - 13) * 30 + 345) for i in
                                range(1, 13)}

                # Calculate vector components
                sum_sin = 0
                sum_cos = 0
                total_precip = 0
                for _, row in group.iterrows():
                    month = row['month']
                    precip = row['precipitation']
                    angle = month_angles[month]
                    sum_sin += precip * np.sin(angle)
                    sum_cos += precip * np.cos(angle)
                    total_precip += precip

                # Handle years with no precipitation
                if total_precip == 0:
                    results.append({'year': year, 'pcp_angle': np.nan, 'pcp_month': np.nan})
                    continue

                # Calculate resultant vector angle
                pcp_rad = np.arctan2(sum_sin, sum_cos)
                if pcp_rad < 0:
                    pcp_rad += 2 * np.pi  # Convert to [0, 2π]

                # Convert angle to degrees
                pcp_angle = degrees(pcp_rad)

                # Calculate PCP month
                pcp_month = ((pcp_angle - 345) % 360) / 30 + 1
                if pcp_month > 12:
                    pcp_month -= 12

                # Save results
                results.append({
                    'year': year,
                    'pcp_angle': pcp_angle,
                    'pcp_month': pcp_month
                })

            except Exception as e:
                print(f"Failed to process year {year} in file {filename}: {str(e)}")
                continue

        # Save station results
        if len(results) > 0:
            df_pcp = pd.DataFrame(results)
            output_path = os.path.join(output_dir, filename)
            df_pcp.to_csv(output_path, index=False)
            print(f"Generated results for {filename}")
        else:
            print(f"No valid results for file {filename}")


# Example usage
input_dir = r"F:\MPCID\Station\monthlydate"
output_dir = r"F:\MPCID\Station\PCP\PCP_result_monthly"

calculate_pcp_station_monthly(input_dir, output_dir)
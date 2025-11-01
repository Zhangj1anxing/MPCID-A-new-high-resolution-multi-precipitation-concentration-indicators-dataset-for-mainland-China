import pandas as pd
import numpy as np
import os


def calculate_pci(data):
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Invalid data. Please, check your input object.")

    # Check if columns are as expected
    if not all(col in data.columns for col in ['year', 'month', 'precipitation']):
        raise ValueError("Data must contain 'year', 'month', and 'precipitation' columns.")

    # Calculate p1 and p2
    p1 = data.groupby('year')['precipitation'].apply(lambda x: (x ** 2).sum())
    p2 = data.groupby('year')['precipitation'].sum()

    # Calculate PCI
    pci = 100 * (p1 / (p2 ** 2))

    # Create result DataFrame
    result = pd.DataFrame({'year': pci.index, 'pci': pci.values})

    return result


def process_files(input_directory, output_file):
    results = []

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_directory, filename)
            data = pd.read_csv(filepath)

            pci_result = calculate_pci(data)
            pci_result['filename'] = os.path.splitext(filename)[0]  # Add filename without extension

            results.append(pci_result)

    # Concatenate all results and save to CSV
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(output_file, index=False)


# Example usage
input_directory = r' '  # Update this path
output_file = r' '  # Update this path
process_files(input_directory, output_file)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from matplotlib.offsetbox import AnchoredText


def preprocess_data(df):
    """
    Classify daily precipitation into 1mm intervals and calculate PM, ni, Pi.
    """
    max_value = df['Value'].max()
    classes = np.arange(0, max_value + 1, 1)
    results = []
    for i in range(len(classes) - 1):
        lower, upper = classes[i], classes[i + 1]
        mask = (df['Value'] > lower) & (df['Value'] <= upper)
        PM = (lower + upper) / 2  # Midpoint of the interval
        ni = mask.sum()  # Number of days in the interval
        Pi = df.loc[mask, 'Value'].sum()  # Total precipitation in the interval
        results.append([PM, ni, Pi])
    results_df = pd.DataFrame(results, columns=['PM', 'ni', 'Pi'])
    results_df = results_df[results_df['ni'] > 0]  # Remove intervals with no precipitation
    results_df['Cumulative_ni'] = results_df['ni'].cumsum()  # Cumulative number of days
    results_df['Cumulative_Pi'] = results_df['Pi'].cumsum()  # Cumulative precipitation
    total_ni = results_df['ni'].sum()
    total_Pi = results_df['Pi'].sum()
    results_df['ni%'] = results_df['Cumulative_ni'] / total_ni * 100  # Cumulative days percentage
    results_df['Pi%'] = results_df['Cumulative_Pi'] / total_Pi * 100  # Cumulative precipitation percentage
    return results_df


def fit_and_calculate_ci(df, year, save_dir):
    """
    Fit exponential curve and calculate CI (Climate Imprint).
    """

    def exp_function(X, b, c):
        return X * np.exp(-b * (100 - X) ** c)

    x_data = df['ni%']
    y_data = df['Pi%']
    # Curve fitting with bounds
    popt, pcov = curve_fit(exp_function, x_data, y_data, bounds=(0, [np.inf, 2]))
    b, c = popt

    # Generate fitted curve
    x_new = np.linspace(0, 100, 500)
    y_fit = exp_function(x_new, b, c)

    # Calculate area using Simpson's rule
    area_under_curve = simpson(y_fit, x_new)
    area_total = 5000  # Area under the equidistribution line (100x100/2)
    CI = (area_total - area_under_curve) / area_total  # Climate Imprint

    # Calculate R-squared
    y_pred = exp_function(x_data, b, c)
    R2 = np.corrcoef(y_data, y_pred)[0, 1] ** 2

    # Save calculation process file with numerical formatting
    process_file = os.path.join(save_dir, f"CI_Process_{year}.csv")
    df.to_csv(process_file, index=False, float_format='%.4f')

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'bo-', label='Observed Data')
    plt.plot(x_new, y_fit, 'r-', label=f'Fitted Curve: $Y = X \\cdot e^{{-b(100-X)^c}}$\n$b={b:.4f}$, $c={c:.4f}$')
    plt.plot(x_new, x_new, 'g--', label='Equidistribution Line')
    plt.fill_between(x_new, x_new, y_fit, color='gray', alpha=0.3, label='Area Difference')

    plt.title(f"Precipitation Concentration Index (Year: {year})", fontsize=14)
    plt.xlabel("Cumulative Rainy Days (%)", fontsize=12)
    plt.ylabel("Cumulative Rainfall (%)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Add CI and R-squared information
    anchored_text = AnchoredText(f"CI (Climate Imprint): {CI:.4f}\n$R^2$: {R2:.4f}",
                                loc='lower right', prop=dict(size=10))
    plt.gca().add_artist(anchored_text)

    # Save plot
    image_file = os.path.join(save_dir, f"CI_Plot_{year}.png")
    plt.savefig(image_file, dpi=300)
    plt.close()

    return CI, R2


def process_all_files(source_dir, save_dir):
    """
    Process all .csv files in the source_dir and store results in save_dir.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_name in os.listdir(source_dir):
        if file_name.endswith('.csv'):
            station_name = os.path.splitext(file_name)[0]  # Remove file extension
            source_file = os.path.join(source_dir, file_name)
            station_save_dir = os.path.join(save_dir, station_name)

            if not os.path.exists(station_save_dir):
                os.makedirs(station_save_dir)

            print(f"Processing {file_name}...")

            # Load data
            data = pd.read_csv(source_file, parse_dates=['Date'])
            data['Year'] = data['Date'].dt.year

            ci_results = []
            for year, group in data.groupby('Year'):
                processed_df = preprocess_data(group)
                CI, R2 = fit_and_calculate_ci(processed_df, year, station_save_dir)
                ci_results.append([year, CI, R2])

            # Save annual CI results
            ci_results_df = pd.DataFrame(ci_results, columns=['Year', 'CI (Climate Imprint)', 'R^2'])
            ci_results_file = os.path.join(station_save_dir, f"{station_name}_CI_Results.csv")
            ci_results_df.to_csv(ci_results_file, index=False)
            print(f"Results saved for {file_name} in {station_save_dir}")


# Usage
source_dir = r" "  # Update source directory path
save_dir = r" "        # Update save directory path

process_all_files(source_dir, save_dir)
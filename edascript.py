import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# --- Configuration ---
# Define the path to your cleaned data file
input_folder = 'preprocessed data.daml'
input_filename = 'cleandata.csv'
input_path = os.path.join(input_folder, input_filename)

# Define the folder where plots will be saved
plots_output_folder = 'eda_plots'
os.makedirs(plots_output_folder, exist_ok=True) # Create the plots folder if it doesn't exist
print(f"Attempting to create/ensure directory: {os.path.abspath(plots_output_folder)}") # <--- ADD THIS LINE

# Set a style for plots for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

# ... (rest of your code remains the same) ...

# --- Configuration ---
# Define the path to your cleaned data file
input_folder = 'preprocessed data.daml'
input_filename = 'cleandata.csv'
input_path = os.path.join(input_folder, input_filename)

# Define the folder where plots will be saved
plots_output_folder = 'eda_plots'
os.makedirs(plots_output_folder, exist_ok=True) # Create the plots folder if it doesn't exist

# Set a style for plots for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

# --- 1. Load the Cleaned Data ---
print("--- 1. Loading Cleaned Data ---")
try:
    df = pd.read_csv(input_path)
    print(f"Cleaned data loaded successfully from: {input_path}")
    print("\nFirst 5 rows of the cleaned data:")
    print(df.head())
    print("\nInfo of the cleaned data:")
    print(df.info())

    # Ensure 'Date' column is in datetime format if it wasn't saved as such
    # (Sometimes CSV saving/loading can convert it back to object/string)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True) # Drop rows where date conversion failed again

except FileNotFoundError:
    print(f"Error: The file '{input_path}' was not found.")
    print("Please ensure the data cleaning script was run successfully and 'cleandata.csv' exists in 'preprocessed data.daml'.")
    exit() # Exit the script if the file is not found

# Identify numerical columns for EDA
# We include all price-related columns and the date components we created
numerical_cols = [
    'Minimum Price', 'Maximum Price', 'Average Price',
    'Price Range', 'Price_Difference',
    'Date_Year', 'Date_Month', 'Date_Day'
]

# Check if Date_Year and Date_Month columns exist, and add them if they do
# This is a safeguard in case the previous cleaning script run didn't include them
# If they are missing, please re-run the *latest* data cleaning script.
if 'Date_Year' not in df.columns:
    print("\nWarning: 'Date_Year' column not found. Ensuring it's created for EDA.")
    df['Date_Year'] = df['Date'].dt.year
if 'Date_Month' not in df.columns:
    print("Warning: 'Date_Month' column not found. Ensuring it's created for EDA.")
    df['Date_Month'] = df['Date'].dt.month

# Filter numerical_cols to only include columns actually present in the DataFrame
numerical_cols = [col for col in numerical_cols if col in df.columns]

# --- 2. Statistical Summary (Mean, Median, Std Dev, Min, Max, Quartiles) ---
print("\n--- 2. Statistical Summary for Numerical Columns ---")
# .describe() provides count, mean, std, min, 25%, 50% (median), 75%, max
print(df[numerical_cols].describe())

# --- 3. IQR Calculation ---
print("\n--- 3. Interquartile Range (IQR) for Numerical Columns ---")
for col in numerical_cols:
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    # Calculate IQR
    IQR = Q3 - Q1
    print(f"{col}: IQR = {IQR:.2f} (Q1={Q1:.2f}, Q3={Q3:.2f})")

# --- 4. Histograms ---
print("\n--- 4. Generating Histograms ---")
for col in numerical_cols:
    plt.figure(figsize=(10, 6)) # Set figure size for better readability
    sns.histplot(df[col], kde=True, bins=30, color='skyblue') # kde=True adds a density curve
    plt.title(f'Histogram of {col}', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6) # Add a subtle grid
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(os.path.join(plots_output_folder, f'{col}_histogram.png'))
    plt.close() # Close the plot to free up memory

print(f"Histograms saved to '{plots_output_folder}' folder.")

# --- 5. Box Plots ---
print("\n--- 5. Generating Box Plots ---")
for col in numerical_cols:
    plt.figure(figsize=(8, 6)) # Set figure size
    sns.boxplot(y=df[col], color='lightcoral') # Box plot for a single variable
    plt.title(f'Box Plot of {col}', fontsize=16)
    plt.ylabel(col, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, f'{col}_boxplot.png'))
    plt.close() # Close the plot

print(f"Box plots saved to '{plots_output_folder}' folder.")

print("\n--- EDA Complete ---")
print(f"All statistical summaries are printed above. Visualizations are saved in the '{plots_output_folder}' folder.")
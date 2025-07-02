import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the dataset, specifying no header as the first row is data, not headers
df = pd.read_csv('Kalimati_Tarkari_Price.csv', header=None)

# Rename the columns to something meaningful
df.columns = ['Commodity', 'Date', 'Unit', 'Minimum Price', 'Maximum Price', 'Average Price']

# Convert price columns to numeric types. 'errors=coerce' will turn any non-numeric values into NaN (Not a Number)
df['Minimum Price'] = pd.to_numeric(df['Minimum Price'], errors='coerce')
df['Maximum Price'] = pd.to_numeric(df['Maximum Price'], errors='coerce')
df['Average Price'] = pd.to_numeric(df['Average Price'], errors='coerce')

# Fill any NaN values in price columns with the mean of that column.
# This ensures no missing data before converting to integer.
df['Minimum Price'] = df['Minimum Price'].fillna(df['Minimum Price'].mean()).astype(int)
df['Maximum Price'] = df['Maximum Price'].fillna(df['Maximum Price'].mean()).astype(int)
df['Average Price'] = df['Average Price'].fillna(df['Average Price'].mean()).astype(int) # Convert Average Price to integer as requested

# Convert the 'Date' column to datetime objects for easier manipulation
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove any rows where the 'Date' could not be converted (e.g., corrupted date entries)
df.dropna(subset=['Date'], inplace=True)

# Create the three new columns as requested:
# 1. 'Price Range': The difference between the maximum and minimum prices
# 2. 'Price_Difference': The difference between the average and minimum prices
# 3. 'Date_Day': Extracts the day number from the 'Date' column
df['Price Range'] = df['Maximum Price'] - df['Minimum Price']
df['Price_Difference'] = df['Average Price'] - df['Minimum Price']
df['Date_Day'] = df['Date'].dt.day

# Define the path where the cleaned data will be saved
output_folder = 'preprocessed data.daml'
output_filename = 'cleandata.csv'
output_path = os.path.join(output_folder, output_filename) # Use os.path.join for cross-platform compatibility

# Create the output folder if it doesn't already exist
os.makedirs(output_folder, exist_ok=True)

# Save the cleaned DataFrame to a new CSV file without the DataFrame index
df.to_csv(output_path, index=False)

print(f"Cleaned data successfully saved to: {output_path}")

df_cleaned = df.copy()

def preprocess_data(df=df_cleaned):
    from sklearn.model_selection import train_test_split

    # Feature selection - adjust these based on your needs
    features = ['Minimum Price', 'Maximum Price', 'Price Range', 'Price_Difference', 'Date_Day']
    X = df[features]
    
    # Regression target (continuous value)
    y_reg = df['Average Price']
    
    # Classification target (categorical - example: price ranges)
    # Create 3 price categories (low, medium, high)
    y_clf = pd.cut(df['Average Price'], 
                  bins=3, 
                  labels=['low', 'medium', 'high'])
    
    # Split data for regression and classification
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42)
    
    _, _, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf
import os
import pandas as pd
import numpy as np


# Load the dataset, specifying no header
df = pd.read_csv('kalimati-tarkari-prices-from-may-2021-to-september-2023.csv', header=None)

# Rename the columns
df.columns = ['Vegetables', 'Date', 'Unit', 'Minimum Price', 'Maximum Price', 'Average Price']

# Convert price columns to numeric types. 'errors=coerce' will turn any non-numeric values into NaN (Not a Number)
df['Minimum Price'] = pd.to_numeric(df['Minimum Price'], errors='coerce')
df['Maximum Price'] = pd.to_numeric(df['Maximum Price'], errors='coerce')
df['Average Price'] = pd.to_numeric(df['Average Price'], errors='coerce')

# Fill any NaN values in price columns with the mean of that column.
# This ensures no missing data before converting to integer.
df['Minimum Price'] = df['Minimum Price'].fillna(df['Minimum Price'].mean())
df['Maximum Price'] = df['Maximum Price'].fillna(df['Maximum Price'].mean())
df['Average Price'] = df['Average Price'].fillna(df['Average Price'].mean())

# Convert price columns to integer
df['Minimum Price'] = df['Minimum Price'].astype(int)
df['Maximum Price'] = df['Maximum Price'].astype(int)
df['Average Price'] = df['Average Price'].astype(int) # Convert Average Price to integer as requested

# Convert the 'Date' column to datetime objects for easier manipulation
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove any rows where the 'Date' could not be converted (e.g., corrupted date entries)
df.dropna(subset=['Date'], inplace=True)

print("\n--- Part 3: Data Type Conversion and Missing Value Handling ---")
print("First 5 rows after type conversion and handling missing values:")
print(df.head())
print("\nInfo after type conversion and handling missing values:")
print(df.info())


# Create the three new columns as requested:
df['Price Range'] = df['Maximum Price'] - df['Minimum Price']
df['Price_Difference'] = df['Average Price'] - df['Minimum Price']

# --- NEW COLUMNS ADDED HERE ---
df['Date_Year'] = df['Date'].dt.year
df['Date_Month'] = df['Date'].dt.month
df['Date_Day'] = df['Date'].dt.day
# --- END OF NEW COLUMNS ---

# ======FEATURE ENGINEERING======
# Add more meaningful temporal features
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Day_of_Year'] = df['Date'].dt.dayofyear
df['Is_Weekend'] = df['Day_of_Week'].isin([5,6]).astype(int)
df['Season'] = (df['Date'].dt.month % 12 + 3) // 3  # 1=winter, 2=spring, etc.

# Add vegetable-specific features
veg_stats = df.groupby('Vegetables')['Average Price'].agg(['mean', 'std']).reset_index()
veg_stats.columns = ['Vegetables', 'Veg_Mean_Price', 'Veg_Std_Price']
df = pd.merge(df, veg_stats, on='Vegetables', how='left')

# Add lag features for time series analysis
df['Price_Change_1D'] = df.groupby('Vegetables')['Average Price'].diff(1)
df['Price_Change_7D'] = df.groupby('Vegetables')['Average Price'].diff(7)
# ====== END OF NEW FEATURE ENGINEERING ======

print("\n--- Part 4: Feature Engineering (New Columns) ---")
print("First 5 rows after creating new features (including Year, Month, Day):")
print(df.head())
print("\nInfo after creating new features:")
print(df.info())

# Define the output folder and filename
output_folder = 'preprocessed data.daml'
output_filename = 'cleandata.csv'
output_path = f"{output_folder}/{output_filename}"

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the cleaned dataset
df.to_csv(output_path, index=False)

print(f"Cleaned data saved to: {output_path}")

df_cleaned = df.copy()

def preprocess_data(df=df_cleaned):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # Feature selection - adjust these based on your needs
    features = ['Date_Year', 'Date_Month', 'Date_Day',
        'Day_of_Week', 'Day_of_Year', 'Is_Weekend', 'Season',
        'Veg_Mean_Price', 'Veg_Std_Price',
        'Price_Change_1D', 'Price_Change_7D']
    X = df[features]
    
    # Regression target (continuous value)
    y_reg = df['Average Price']
    
    # Classification target (categorical - example: price ranges)
    # Create 3 price categories (low, medium, high)
    y_clf = pd.qcut(df['Average Price'], 
                  q=3, 
                  labels=['low', 'medium', 'high'])
    
    # Split data for regression and classification
    # For Regression
    X_train, X_temp_reg, y_train_reg, y_temp_reg = train_test_split(
        X, y_reg, test_size=0.4, random_state=42)  # 60% train, 40% temp
    X_val_reg, X_test_reg, y_val_reg, y_test_reg = train_test_split(
        X_temp_reg, y_temp_reg, test_size=0.5, random_state=42)  # Split temp into 20% val, 20% test
    
    # For Classification (using same indices to maintain alignment)
    X_train,X_temp_clf, y_train_clf,y_temp_clf = train_test_split(
        X, y_clf, test_size=0.4, random_state=42)  # Same test_size as regression
    X_val_clf,X_test_clf, y_val_clf, y_test_clf = train_test_split(
        X_temp_clf, y_temp_clf, test_size=0.5, random_state=42)
    
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val_reg = imputer.transform(X_val_reg)
    X_test_reg = imputer.transform(X_test_reg)
    X_val_clf = imputer.transform(X_val_clf)
    X_test_clf = imputer.transform(X_test_clf)

    #Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_reg_scaled = scaler.transform(X_val_reg)
    X_test_reg_scaled = scaler.transform(X_test_reg)
    X_val_clf_scaled = scaler.transform(X_val_clf)
    X_test_clf_scaled = scaler.transform(X_test_clf)

    return(    # Scaled features
        X_train_scaled,
        X_val_reg_scaled, X_test_reg_scaled,
        X_val_clf_scaled, X_test_clf_scaled,
        y_train_reg, y_val_reg, y_test_reg,
        y_train_clf, y_val_clf, y_test_clf,
        scaler,
        features
    )
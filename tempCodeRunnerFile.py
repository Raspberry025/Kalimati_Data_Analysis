#Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib . pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from cleandata import preprocess_data, df_cleaned
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error , mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


#-----Regression Model-----
def load_and_preprocess():
    return preprocess_data(df_cleaned)

def regression_analysis(X_train, X_test, y_train_reg, y_test_reg):
    print("\n" + '='*40)
    print("Regression Analysis (SVR)")
    print("="*40)

    #Training SVR
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train, y_train_reg)
    y_pred = svr.predict(X_test)

    #Calculating Metrics
    mae = mean_absolute_error ( y_test_reg , y_pred)
    r2 = r2_score ( y_test_reg , y_pred )
    rmse = np.sqrt ( mean_squared_error ( y_test_reg , y_pred ) )

    print ( f" R2 = {r2 :.3f}")
    print ( f" RMSE = { rmse :.3f}")
    print ( f"MAE = {mae :.3f}")

    # P-values calculation
    X_with_const = np.column_stack([np.ones(X_train.shape[0]), X_train])
    model = sm.OLS(y_train_reg, X_with_const)
    results = model.fit()

    print("\nFeature P-values: ")
    pvalues = results.pvalues[1:] #Skip intercept
    for idx, pval in enumerate(pvalues):
        print(f"Feature {idx}: {pval:.4f}{' *' if pval < 0.05 else ''}")
    
    # Plotting
    plt.figure(figsize=(10,5))
    plt.scatter(y_test_reg, y_pred, alpha=0.5)
    plt.plot([y_test_reg.min(), y_test_reg.max()],
             [y_test_reg.min(), y_test_reg.max()], 'r--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("SVR: Actual VS Predicted")
    plt.show()

    return svr

#------Classification------
def classification_analysis(X_train, X_test, y_train_clf, y_test_clf):
    print("\n" + "="*40)
    print("CLASSIFICATION ANALYSIS (RANDOM FOREST)")
    print("="*40)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state= 42)
    rf.fit(X_train, y_train_clf)
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_clf, y_pred)
    report = classification_report(y_test_clf, y_pred,target_names=['low', 'medium', 'high'])
    
    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_clf, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['low', 'medium', 'high'],
                yticklabels=['low', 'medium', 'high'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Feature Importance
    importances = rf.feature_importances_
    plt.figure(figsize=(10,5))
    plt.barh(range(X_train.shape[1]), importances, align='center')
    plt.yticks(range(X_train.shape[1]), [f'Feature {i}' for i in range(X_train.shape[1])])
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.show()
    
    return rf

#----Main Execution-----
if __name__ == "__main__":
    # Load and pre-process
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = load_and_preprocess()

    # Verify data shapes
    print("=== DATA SHAPES ===")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Regression targets - y_train: {y_train_reg.shape}, y_test: {y_test_reg.shape}")
    print(f"Classification targets - y_train: {y_train_clf.shape}, y_test: {y_test_clf.shape}")
    
    # Run analyses
    svr_model = regression_analysis(X_train, X_test, y_train_reg, y_test_reg)
    rf_model = classification_analysis(X_train, X_test, y_train_clf, y_test_clf)

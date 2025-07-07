#Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import KFold, cross_val_score
from cleandata import preprocess_data, df_cleaned
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error , mean_absolute_error, accuracy_score, classification_report, confusion_matrix


#-----Regression Model-----
def load_and_preprocess():
    return preprocess_data(df_cleaned)

def regression_analysis(X_train, X_test, y_train_reg, y_test_reg):
    print("\n" + '='*40)
    print("Regression Analysis (RFG)")
    print("="*40)

     # Parameter distributions for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(10, 100),       # Number of trees
        'max_depth': [None, 5, 10],        # Tree depth
        'min_samples_split': randint(2, 10),     # Minimum samples to split
        'min_samples_leaf': randint(1, 10),      # Minimum samples at leaf
        'max_features': ['sqrt']   # Feature selection
    }

    #Initializing K-fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    #Training SVR
    rfg = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,           # Number of parameter settings sampled
        cv=kfold,             
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    rfg.fit(X_train, y_train_reg)
    best_rfg = rfg.best_estimator_

    #Cross-validated scores (for training data)
    cv_scores = cross_val_score(
        best_rfg,
        X_train,
        y_train_reg,
        cv=kfold,
        scoring='neg_mean_squared_error'
    )
    print(f"\nCross-validated RMSE: {-cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

    y_pred = best_rfg.predict(X_test)

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
    plt.title("RandomForestRegressor: Actual VS Predicted")
    plt.show()

    return best_rfg

#------Classification------
def classification_analysis(X_train, X_test, y_train_clf, y_test_clf):
    print("\n" + "="*40)
    print("CLASSIFICATION ANALYSIS (RANDOM FOREST)")
    print("="*40)

    param_dist = {
        'n_estimators': randint(50, 300),        # Wider range but random sampling
        'max_depth': [None] + list(range(5, 50, 5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt'],  # Feature subsampling
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced']       # Handle class imbalance
    }
    # Initialize K-fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # Train Random Forest
    rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=30,               # More iterations for complex model
        cv=kfold,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rf.fit(X_train, y_train_clf)
    best_rf = rf.best_estimator_
    #Cross-validated scores (for training data)
    cv_scores = cross_val_score(
        best_rf,
        X_train,
        y_train_clf,
        cv=kfold,
        scoring='accuracy'
    )
    print(f"\nCross-validated Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

    y_pred = best_rf.predict(X_test)
    
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
    importances = best_rf.feature_importances_
    plt.figure(figsize=(10,5))
    plt.barh(range(X_train.shape[1]), importances, align='center')
    plt.yticks(range(X_train.shape[1]), [f'Feature {i}' for i in range(X_train.shape[1])])
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.show()
    
    return best_rf

#----Main Execution-----
if __name__ == "__main__":
    # Load and pre-process
    (X_train_scaled,
        X_val_reg_scaled, X_test_reg_scaled,
        X_val_clf_scaled, X_test_clf_scaled,
        y_train_reg, y_val_reg, y_test_reg,
        y_train_clf, y_val_clf, y_test_clf,
        scaler,
        features) = load_and_preprocess()
    
    # Run analyses
    rfg_model = regression_analysis(X_train_scaled, X_test_reg_scaled, y_train_reg, y_test_reg)
    rf_model = classification_analysis(X_train_scaled, X_test_clf_scaled, y_train_clf, y_test_clf)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv(r"C:\Users\mudda\Downloads\archive (26)\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Clean data
df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# Target variable processing
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
y = df['Attrition']
X = df.drop('Attrition', axis=1)

# Categorical column processing
cat_cols = X.select_dtypes(include=['object', 'category']).columns

# Train test split FIRST to prevent any data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# It's better to compute medians/modes on the TRAINING set only
imputation_values = {}
for col in X_train.select_dtypes(include=['int64', 'float64']).columns:
    imputation_values[col] = X_train[col].median()
for col in cat_cols:
    imputation_values[col] = X_train[col].mode()[0]

# Save the imputation values for the prediction endpoint
joblib.dump(imputation_values, "imputation_values.pkl")

# Save the raw training columns before get_dummies (if useful for frontend)
joblib.dump(list(X.columns), "raw_columns.pkl")


# Process categorical variables. Get dummies only on X_train, then align X_test.
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Align columns to ensure test set has exactly same columns as train set
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Save the encoded columns for the backend prediction endpoint
joblib.dump(list(X_train_encoded.columns), "columns.pkl")


print("Creating pipeline with StandardScaler -> SMOTE -> GradientBoostingClassifier")
# Create a proper pipeline to avoid data leakage during cross-validation
# Scale data -> Apply SMOTE -> Train model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Define hyperparameter distribution
# Note the prefix 'classifier__' to target the estimator inside the pipeline
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [2, 3, 4],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__subsample': [0.8, 0.9, 1.0]
}

# Run RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20, # reduced to speed up execution
    scoring='f1',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Starting hyperparameter tuning...")
random_search.fit(X_train_encoded, y_train)

# Output best parameters
print("\nBest Parameters found:", random_search.best_params_)

# Evaluate the best pipeline on the UNTOUCHED test set
best_model = random_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test_encoded)
y_prob = best_model.predict_proba(X_test_encoded)[:, 1]

print("\n--- FINAL MODEL CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# Get metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save the full pipeline as the model 
# (This means incoming prediction data must be one-hot encoded, but NOT manually scaled)
joblib.dump(best_model, "model.pkl")
print("Saved best_model pipeline to model.pkl")
print("Saved expected features to columns.pkl")
print("Saved imputation medians/modes to imputation_values.pkl")

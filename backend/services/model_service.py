import joblib
import pandas as pd
import numpy as np
import threading
import os

_model = None
_columns = None
_imputation_values = None
_lock = threading.Lock()

def get_model_artifacts():
    global _model, _columns, _imputation_values
    if _model is None:
        with _lock:
            if _model is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                print(f"Loading model artifacts from {base_dir}...")
                _model = joblib.load(os.path.join(base_dir, 'model.pkl'))
                _columns = joblib.load(os.path.join(base_dir, 'columns.pkl'))
                _imputation_values = joblib.load(os.path.join(base_dir, 'imputation_values.pkl'))
                print("Model artifacts loaded.")
    return _model, _columns, _imputation_values

def predict_employee(features: dict):
    '''
    Takes a dict of features, returns probability of attrition.
    Fixes the feature mismatch by one-hot encoding the input to match the training data.
    '''
    model, columns, imputation_values = get_model_artifacts()
    
    # Fill missing features with the learned medians & modes
    full_features = imputation_values.copy()
    
    # Map OverTime from 0/1 (if sent that way) to No/Yes
    if 'OverTime' in features:
        val = features['OverTime']
        if val == 1: features['OverTime'] = 'Yes'
        elif val == 0: features['OverTime'] = 'No'
        
    full_features.update(features)
    
    # Create the DataFrame
    df = pd.DataFrame([full_features])

    # Pre-cleaning: remove target-leakage or meta-cols if present
    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # ONE-HOT ENCODING (Matches the training process)
    # We use only the categorical columns defined during training (indirectly via get_dummies on original df)
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    # Filter to only use cat_cols that are actually in df
    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    
    df_encoded = pd.get_dummies(df, columns=existing_cat_cols)

    # REINDEX to ensure exact column alignment with model's expected features
    # Columns missing in the input (dummy variables not triggered) are filled with 0
    df_final = df_encoded.reindex(columns=columns, fill_value=0)

    # Convert all columns to numeric (bool dummies to 0/1)
    df_final = df_final.astype(float)

    # predict using the pipeline
    # Note: best_model in train_improved_model.py is a Pipeline [Scaler, SMOTE, Classifier]
    # Scaling is handled automatically by the first step of the pipeline.
    prob = model.predict_proba(df_final)[0][1]

    return float(prob)
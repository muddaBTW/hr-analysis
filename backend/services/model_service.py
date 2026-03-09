import joblib
import pandas as pd

import threading

_model = None
_columns = None
_imputation_values = None
_lock = threading.Lock()

def get_model_artifacts():
    global _model, _columns, _imputation_values
    if _model is None:
        with _lock:
            if _model is None:
                print("Loading model artifacts...")
                _model = joblib.load('model.pkl')
                _columns = joblib.load('columns.pkl')
                _imputation_values = joblib.load('imputation_values.pkl')
                print("Model artifacts loaded.")
    return _model, _columns, _imputation_values

def predict_employee(features: dict):
    '''
    Takes a dict of features, returns probability of attrition.
    Uses the ML Pipeline which handles its own scaling and encoding.
    '''
    model, columns, imputation_values = get_model_artifacts()
    
    # Fill missing features with the learned medians & modes
    # This ensures we have the exact set of columns the model first saw (raw)
    full_features = imputation_values.copy()
    full_features.update(features)
    
    # Create the DataFrame with the RAW columns (as the pipeline expects)
    # Note: 'columns' here refers to the raw input columns before the pipeline's transformer
    df = pd.DataFrame([full_features])

    # Ensure all columns exist in the right order (raw columns)
    # We should use the original column names from the training set
    raw_columns = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber',
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
        'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18',
        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]
    df = df.reindex(columns=raw_columns)

    # predict using the pipeline
    # The pipeline handles scaling and encoding internally!
    prob = model.predict_proba(df)[0][1]

    return float(prob)
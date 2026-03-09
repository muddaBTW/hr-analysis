import joblib
import pandas as pd

_model = None
_columns = None
_imputation_values = None

def get_model_artifacts():
    global _model, _columns, _imputation_values
    if _model is None:
        print("Loading model artifacts...")
        _model = joblib.load('model.pkl')
        _columns = joblib.load('columns.pkl')
        _imputation_values = joblib.load('imputation_values.pkl')
        print("Model artifacts loaded.")
    return _model, _columns, _imputation_values

def predict_employee(features: dict):
    '''
    takes dict as a input
    returns probability of attrition
    '''
    model, columns, imputation_values = get_model_artifacts()
    
    # Fill missing features with the learned medians & modes
    full_features = imputation_values.copy()
    full_features.update(features)
    
    # Create the DataFrame
    df = pd.DataFrame([full_features])

    # Convert categories to one-hot encoding
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Reindex to ensure it precisely matches the columns the model was trained on
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    # predict
    prob = model.predict_proba(df_encoded)[0][1]

    return float(prob)
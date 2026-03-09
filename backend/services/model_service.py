import joblib
import pandas as pd

model = joblib.load('model.pkl')
columns = joblib.load('columns.pkl')
# Load the medians/modes calculated from training data
imputation_values = joblib.load('imputation_values.pkl')

def predict_employee(features: dict):
    '''
    takes dict as a input
    returns probability of attrition
    '''
    # Fill missing features with the learned medians & modes
    full_features = imputation_values.copy()
    full_features.update(features)
    
    # Create the DataFrame
    df = pd.DataFrame([full_features])

    # Convert categories to one-hot encoding
    # get_dummies might only produce columns for the categories present in this single row
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Reindex to ensure it precisely matches the columns the model was trained on
    # Since we imputed continuous variables, fill_value=0 is now ONLY applying to 
    # one-hot encoded dummy variables that were missing from this row, which is the mathematically correct behavior.
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    # predict
    prob = model.predict_proba(df_encoded)[0][1]

    return float(prob)
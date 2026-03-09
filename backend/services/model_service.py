import joblib
import pandas as pd

model = joblib.load('model.pkl')
columns = joblib.load('columns.pkl')

def predict_employee(features:dict):
    '''
    takes dict as a input
    returns probability of attrition
    '''

    df = pd.DataFrame([features])

    # match to the columns
    df = df.reindex(columns=columns, fill_value=0)

    # predict
    prob = model.predict_proba(df)[0][1]

    return float(prob)
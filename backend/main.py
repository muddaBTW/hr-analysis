from fastapi import FastAPI
from schema.prediction_schema import Employee
from services.model_service import predict_employee
from schema.rag_schema import Question
from services.rag_service import ask_question

app = FastAPI()

@app.get('/')
def home():
    return {'message':'Backend Running'}

@app.post('/predict')
def predict(data:Employee):
    probability = predict_employee(data.features)

    return {
        'probability': probability,
        'risk':'High' if probability > 0.5 else 'Low'
    }

@app.post('/ask')
def ask(data:Question):
    answer = ask_question(data.query, data.api_key)

    return {
        'answer':answer
    }
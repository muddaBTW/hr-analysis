from fastapi import FastAPI
import os

app = FastAPI()

@app.get('/')
def home():
    print("Health check ping received at /")
    return {'message': 'Backend Running', 'status': 'healthy'}

@app.post('/predict')
def predict(data: dict):
    # Lazy import inside the route to keep startup instant
    from services.model_service import predict_employee
    from schema.prediction_schema import Employee
    
    # Validate manually or just pass features
    features = data.get("features", {})
    probability = predict_employee(features)

    return {
        'probability': probability,
        'risk': 'High' if probability > 0.5 else 'Low'
    }

@app.post('/ask')
def ask(data: dict):
    # Lazy import inside the route
    from services.rag_service import ask_question
    
    query = data.get("query", "")
    api_key = data.get("api_key")
    
    answer = ask_question(query, api_key)

    return {
        'answer': answer
    }
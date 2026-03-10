from fastapi import FastAPI
import os
import threading
from contextlib import asynccontextmanager

# Define a lifespan event to trigger background loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs in the background as soon as the server starts
    def load_everything():
        try:
            print("--- Proactive Background Loading Started ---")
            from services.model_service import get_model_artifacts
            get_model_artifacts() # High-priority, lightweight
            print("--- Proactive Background Loading Complete (ML Model) ---")
            
            # Pre-load RAG knowledge base (lightweight TF-IDF, ~5MB)
            from services.rag_service import get_knowledge_index
            get_knowledge_index()
            print("--- Proactive Background Loading Complete (RAG) ---")
        except Exception as e:
            print(f"Background loading failed (will retry on first request): {e}")

    # Start the thread
    thread = threading.Thread(target=load_everything, daemon=True)
    thread.start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get('/')
def home():
    return {'message': 'Backend Running', 'status': 'healthy'}

@app.post('/predict')
def predict(data: dict):
    from services.model_service import predict_employee
    features = data.get("features", {})
    probability = predict_employee(features)
    return {
        'probability': probability,
        'risk': 'High' if probability > 0.5 else 'Low'
    }

@app.post('/ask')
def ask(data: dict):
    from services.rag_service import ask_question
    query = data.get("query", "")
    api_key = data.get("api_key")
    answer = ask_question(query, api_key)
    return {
        'answer': answer
    }
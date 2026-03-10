from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from services.model_service import predict_employee, get_model_artifacts
from services.rag_service import get_retriever, get_rag_chain
import threading
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from root directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path, override=True)

# Define request/response models
class PredictionRequest(BaseModel):
    features: dict

class ChatRequest(BaseModel):
    message: str
    api_key: str = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload model and retriever in the background to speed up first request
    def preload():
        try:
            print("Preloading model artifacts and knowledge index...")
            get_model_artifacts()
            get_retriever()
            print("Preloading complete.")
        except Exception as e:
            print(f"Error during preloading: {e}")

    thread = threading.Thread(target=preload)
    thread.start()
    yield

app = FastAPI(title="HR Attrition AI API", lifespan=lifespan)

@app.post("/predict")
async def predict(request: PredictionRequest):
    probability = predict_employee(request.features)
    risk = "High" if probability > 0.5 else "Low"
    return {"probability": probability, "risk": risk}

@app.post("/chat")
async def chat(request: ChatRequest):
    # If the user provides an API key, we should ideally re-initialize the LLM or pass it through.
    # For now, we'll stick to the server-side key unless we update get_rag_chain.
    # To keep it simple and consistent with the frontend possibility:
    rag_chain = get_rag_chain(api_key=request.api_key)
    response = rag_chain(request.message)
    return {"response": response}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
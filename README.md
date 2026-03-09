# HR Attrition Intelligence 🏢

An AI-powered HR analytics platform to predict and analyze employee attrition using the IBM HR Analytics dataset.

## 🚀 Deployment

This project is designed to be deployed as a split-service architecture:

- **Backend**: FastAPI (Python) - Host on Render.
- **Frontend**: Streamlit (Python) - Host on Streamlit Cloud.

### Project Structure

- `backend/`: FastAPI server, ML models, and RAG logic.
- `frontend/`: Streamlit UI pages.
- `notebook/`: Original data analysis and model training experiments.

### Local Setup

1. Clone the repo.
2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your `GROQ_API_KEY` in a `.env` file in the root.
4. Start the backend:

   ```bash
   cd backend
   python main.py
   ```

5. Start the frontend:

   ```bash
   cd frontend
   streamlit run app.py
   ```

## 🛠️ Features

- **Gradient Boosting Model**: High-accuracy attrition prediction with SMOTE and custom imputation.
- **RAG Chatbot**: AI assistant grounded in HR data using LangChain and Groq.
- **Interactive Dashboard**: Deep EDA and visualization of attrition drivers.

# HR Attrition AI System — Project Component Explanation

This document provides a comprehensive breakdown of every file and component in the **HR Attrition Analysis & Prediction** project.

---

## 📂 Project Structure Overview

```text
hr-ai-system/
├── backend/                # FastAPI Application (Logic & ML)
│   ├── main.py             # API Entry point & Routes
│   ├── model.pkl           # Trained ML Model (Gradient Boosting)
│   ├── columns.pkl         # List of encoded features for the model
│   ├── imputation_values.pkl # Medians/Modes for missing data
│   ├── knowledge.md        # RAG Knowledge base for the AI Chat
│   ├── requirements.txt    # Backend-specific dependencies
│   ├── train_improved_model.py # Script used to train the model
│   └── services/
│       ├── model_service.py # Prediction & Preprocessing logic
│       └── rag_service.py   # AI Chat (RAG) & LLM logic
├── frontend/               # Streamlit Application (UI)
│   └── pages/
│       ├── analysis.py     # EDA Dashboard & Visualizations
│       ├── ai_chat.py      # AI Assistant Interface
│       └── prediction.py   # Employee Attrition Calculator
├── .env                    # Configuration (API Keys)
├── render.yaml             # Cloud Deployment Blueprint
└── requirements.txt        # Full Project Dependencies
```

---

## 🎨 Frontend (Streamlit)

The frontend is built using **Streamlit**, divided into three main functional pages:

### 1. Analysis Dashboard (`frontend/pages/analysis.py`)

- **Purpose**: Visualizes the IBM HR dataset to identify trends and reasons why employees leave.
- **Key Tabs**:
  - **Overview**: High-level KPIs (Total Employees, Attrition Rate, Monthly Income).
  - **Feature Analysis**: Categorical and Numerical plots comparing features (Age, Overtime, etc.) against Attrition.
  - **Dept & Roles**: Breakdown of which departments (Sales vs R&D) and roles (Sales Reps) are most at risk.
  - **Model Performance**: Displays ROC Curves and Precision-Recall curves to show how accurate the AI model is.
  - **Feature Importance**: Shows which factors (like Stock Options and Income) influence the model the most.

### 2. Prediction Engine (`frontend/pages/prediction.py`)

- **Purpose**: A "calculator" where HR can input specific employee details (Age, Income, Distance from Home) to see their probability of leaving.
- **Workflow**: It collects user inputs from the sidebar and sends a POST request to the backend `/predict` endpoint.

### 3. AI Chat Assistant (`frontend/pages/ai_chat.py`)

- **Purpose**: Allows users to ask natural language questions about the HR data (e.g., "What is the average age of employees who leave?").
- **Workflow**: Sends the user's message to the backend `/chat` endpoint and displays the AI's response.

---

## ⚙️ Backend (FastAPI)

The backend provides the "brain" of the application via high-performance API endpoints.

### 1. Main API Entry (`backend/main.py`)

- **Lifespan Manager**: Uses background threads to preload the ML model and the AI knowledge index so that the first user request is fast.
- **Endpoints**:
  - `POST /predict`: Receives employee data, processes it, and returns the attrition risk percentage.
  - `POST /chat`: Receives a message, retrieves relevant HR data using RAG, and generates a response using Groq (Llama 3).
  - `GET /health`: Used by the cloud server to ensure the app is running.

### 2. Model Service (`backend/services/model_service.py`)

- **One-Hot Encoding**: Converts raw text data (like Department names) into the binary format (0/1) required by the ML model.
- **Imputation**: If any data is missing, it automatically fills it with the median/mode from the original training set to prevent errors.
- **Feature Alignment**: Ensures the 35+ input features match the exact order and names the model saw during training.

### 3. RAG Service (`backend/services/rag_service.py`)

- **TF-IDF Retriever**: A lightweight search engine that parses `knowledge.md` to find the most relevant facts based on the user's question.
- **Groq Integration**: Connects to the **Llama-3.3-70b** model via the Groq API to generate human-like answers using only the provided HR context (ensuring no "hallucinations").

---

## 🧠 Data & Machine Learning

### 1. Dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`)

- The core data source containing 1,470 records and 35 features.

### 2. Training Script (`backend/train_improved_model.py`)

- **SMOTE**: Applies the *Synthetic Minority Oversampling Technique* to balance the data (since only ~16% of employees leave, the model needs help learning the "Yes" case).
- **Gradient Boosting**: Uses a high-performance ensemble algorithm for classification.
- **Artifacts**: Saves the model (`model.pkl`), column list (`columns.pkl`), and imputation values for use by the backend.

---

## ☁️ Deployment & Config

- **`render.yaml`**: Configuration file for **Render.com**. It defines how to build the backend (Dockerfile-like steps) and the frontend (Streamlit startup).
- **`.env`**: Contains your **GROQ_API_KEY**. Keeping this in a separate file keeps your secrets safe from being pushed to public logs.
- **`requirements.txt`**: Lists all Python libraries required to run the project (FastAPI, Streamlit, Scikit-learn, etc.).

---

**Summary**: This project is a complete "End-to-End" AI application that turns raw HR data into visual insights, predictive risk scores, and an interactive AI assistant.

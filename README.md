# HR Attrition Intelligence Platform

## Overview

This platform is an AI-powered HR analytics system designed to analyze, predict, and gain insights into employee attrition. It utilizes a gradient boosting prediction model along with a Retrieval-Augmented Generation (RAG) system for AI-driven insights based on comprehensive HR analytics data.

GitHub Repository: [https://github.com/muddaBTW/hr-analysis](https://github.com/muddaBTW/hr-analysis)

## Tech Stack

* **Frontend**: Streamlit (Python)
* **Backend**: FastAPI (Python)
* **AI Model**: Fast, concise AI answers powered by Groq LLM API and Langchain.
* **Machine Learning**: Gradient Boosting Classifier (scikit-learn) with SMOTE for handling imbalanced datasets.
* **RAG Pipeline**: FAISS for vector storage, HuggingFace embeddings (`all-MiniLM-L6-v2`), and Groq (`Llama 3.1 8B Instant`).

## Features

* **AI Chat Assistant**: Provides insightful answers related to HR attrition, stats, and individual features.
* **Analysis Dashboard**: Beautifully styled and highly readable visualizations built with Plotly and Seaborn.
  * Feature Importances (aggregated for clarity)
  * Correlation Heatmaps
  * Demographic and Department-based Breakdowns
* **Prediction Engine**: Accurate predictions configured based on multiple employee attributes.

## Setup Instructions

1. **Clone the Repository**

    ```bash
    git clone https://github.com/muddaBTW/hr-analysis.git
    cd hr-analysis
    ```

2. **Environment Setup**
    Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **API Keys**
    Create a `.env` file in the root directory and add your Groq API key:

    ```env
    GROQ_API_KEY=your_api_key_here
    ```

5. **Run the Backend (FastAPI)**
    Open a terminal, activate the environment, and navigate to the backend folder:

    ```bash
    cd backend
    python -m uvicorn main:app --reload
    ```

6. **Run the Frontend (Streamlit)**
    Open a second terminal, activate the environment, and navigate to the frontend folder:

    ```bash
    cd frontend
    python -m streamlit run app.py
    ```

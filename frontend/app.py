import streamlit as st

st.set_page_config(page_title='HR Attrition Intelligence', layout='wide')

st.title("HR Attrition Intelligence Platform")

st.markdown("""
### AI-Powered Employee Attrition Analysis & Prediction

This platform combines **data analytics**, **machine learning**, and **AI-powered insights**
to help organizations understand, predict, and reduce employee attrition — all in one place.
""")

st.divider()

# ─── Modules ───
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Analysis Dashboard
    Explore interactive visualizations organized in **six tabs**:

    - **Overview** — Attrition distribution pie chart & key facts
    - **Feature Analysis** — Categorical & numerical patterns
    - **Dept & Roles** — Department and job-role-level attrition
    - **Employee Insights** — Satisfaction, work-life balance & income
    - **Correlations** — Feature correlation heatmap
    - **Model Performance** — ROC, Precision-Recall & feature importance
    """)

with col2:
    st.markdown("""
    ### 🔮 Prediction Engine
    Predict individual employee attrition risk in real-time:

    - Input employee details via sidebar controls
    - Get an **attrition probability score** (0–100%)
    - See a clear **High / Low risk** classification
    - Powered by a **Gradient Boosting model** (86% accuracy, 0.80 AUC)
    - Supports proactive HR decision-making
    """)

with col3:
    st.markdown("""
    ### 🤖 AI Chat (RAG)
    Ask natural language questions and get instant, data-backed answers:

    - Attrition drivers & risk factors
    - Dataset statistics & trends
    - Model performance & feature importance
    - HR strategy & retention recommendations
    - Powered by **LangChain + FAISS + Llama 3.1**
    """)

st.divider()

# ─── Dataset ───
st.markdown("""
### 📁 Dataset

**IBM HR Analytics Employee Attrition Dataset** — 1,470 employees × 35 features.

Key metrics: **16.1% attrition rate** · Average age **37 yrs** · Average monthly income **$6,503**

Top predictors: Stock Option Level · Monthly Income · Overtime · Work-Life Balance · Job Satisfaction
""")

st.divider()

# ─── Tech Stack ───
st.markdown("""
### ⚙️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend API | FastAPI |
| ML Model | Gradient Boosting (scikit-learn) + SMOTE |
| AI Chat | LangChain · FAISS · HuggingFace Embeddings · Groq (Llama 3.1 8B) |
| Dataset | IBM HR Analytics (Kaggle) |
""")

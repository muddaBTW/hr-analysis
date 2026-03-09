import streamlit as st
import requests
import os

st.set_page_config(layout="wide")
st.title("Employee Attrition Prediction")

# Use environment variable for deployment, fallback to local for development
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = f"{BACKEND_URL}/predict"

st.sidebar.header("Employee Information")

age = st.sidebar.slider("Age", 18, 60, 30)
daily_rate = st.sidebar.number_input("Daily Rate", 100, 2000, 800)
distance = st.sidebar.slider("Distance From Home", 1, 30, 5)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
total_years = st.sidebar.slider("Total Working Years", 0, 40, 5)
overtime = st.sidebar.selectbox("OverTime (1 = Yes, 0 = No)", [0, 1])
job_level = st.sidebar.slider("Job Level", 1, 5, 2)

features = {
    "Age": age,
    "DailyRate": daily_rate,
    "DistanceFromHome": distance,
    "MonthlyIncome": monthly_income,
    "TotalWorkingYears": total_years,
    "OverTime": overtime,
    "JobLevel": job_level
}

if st.button("Predict Attrition Risk"):
    try:
        response = requests.post(API_URL, json={"features": features})

        if response.status_code == 200:
            result = response.json()

            probability = result["probability"]
            risk = result["risk"]

            st.subheader("Prediction Result")

            if risk == "High":
                st.error(f"High Risk of Attrition ({probability:.1%})")
            else:
                st.success(f"Low Risk of Attrition ({probability:.1%})")

            st.markdown("---")
            st.subheader("Model Performance & Information")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", "81%")
            with col2:
                st.metric("ROC-AUC", "0.81")
            with col3:
                st.metric("Recall (Minority)", "59%")
            with col4:
                st.metric("F1-Score", "0.50")

            st.info("""
            **Model Architecture:** Gradient Boosting Classifier  
            **Preprocessing Hub:** StandardScaler -> SMOTE -> ML Pipeline  
            **Imputation Logic:** Missing inputs are automatically filled with training set medians/modes to ensure stability.
            """)

            st.markdown("""
            ### Interpretation

            The probability represents the model's confidence that the employee may leave.

            Higher probability indicates greater attrition risk.

            HR teams can use this information for proactive retention planning.
            """)

        else:
            st.error("Backend error. Please check if the API server is available.")

    except Exception as e:
        st.error(f"Error processing prediction: {str(e)}")
        st.info(f"Connected to backend at: {BACKEND_URL}")

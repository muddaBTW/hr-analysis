import streamlit as st
import requests

st.set_page_config(layout="wide")
st.title("Employee Attrition Prediction")

API_URL = "http://127.0.0.1:8000/predict"

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
                st.error("High Risk of Attrition")
            else:
                st.success("Low Risk of Attrition")

            st.write(f"Probability of Leaving: {probability:.2%}")

            st.markdown("""
            ### Interpretation

            The probability represents the model's confidence that the employee may leave.

            Higher probability indicates greater attrition risk.

            HR teams can use this information for proactive retention planning.
            """)

        else:
            st.error("Backend error. Please check if the API server is running.")

    except Exception:
        st.error("Could not connect to backend. Ensure FastAPI is running on port 8000.")

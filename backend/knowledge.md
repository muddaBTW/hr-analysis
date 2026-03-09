# HR Attrition Analysis — Comprehensive Knowledge Base

## 1. Project Overview

This is an AI-powered HR analytics platform that analyzes, predicts, and provides insights on employee attrition using the IBM HR Analytics Employee Attrition dataset. The system integrates Exploratory Data Analysis (EDA), Machine Learning (Gradient Boosting with SMOTE), and Retrieval-Augmented Generation (RAG) to deliver actionable HR intelligence.

### Core Modules

1. **Analysis Dashboard** — Interactive visualizations revealing attrition patterns across departments, roles, satisfaction levels, compensation, and tenure.
2. **Prediction Engine** — Real-time ML-based attrition risk scoring for individual employees.
3. **AI Chat Assistant** — Natural language Q&A powered by RAG architecture using FAISS vector search and Groq LLM.

### Technology Stack

- **Frontend**: Streamlit (Python)
- **Backend**: FastAPI (Python)
- **ML Model**: Gradient Boosting Classifier (scikit-learn) with SMOTE for class balancing
- **RAG Pipeline**: LangChain + FAISS + HuggingFace Embeddings (all-MiniLM-L6-v2) + Groq LLM (Llama 3.1 8B Instant)
- **Dataset**: IBM HR Analytics Employee Attrition Dataset from Kaggle

---

## 2. Dataset Details

- **Source**: IBM HR Analytics Employee Attrition Dataset (Kaggle)
- **Total Records**: 1,470 employees
- **Total Features**: 35 columns
- **Target Variable**: Attrition (Binary — Yes or No)

### Attrition Breakdown

- **Yes (Left the company)**: 237 employees (16.1%)
- **No (Stayed)**: 1,233 employees (83.9%)
- *Note: This is a class-imbalanced dataset, which is why SMOTE was applied during model training.*

### Feature Categories

#### Demographics

- **Age**: Range 18 to 60, Mean 36.9 years, Median 36 years
- **Gender**: Male 60%, Female 40%
- **MaritalStatus**: Married 45.6%, Single 32%, Divorced 22.4%
- **Education**: Scale 1-5 (1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor)
- **EducationField**: Life Sciences (41%), Medical (32%), Marketing (11%), Technical Degree (9%), Other (7%)

#### Compensation

- **MonthlyIncome**: Range $1,009 to $19,999, Mean $6,503, Median $4,919 (right-skewed distribution)
- **DailyRate**: Range $102 to $1,499, Mean $802
- **HourlyRate**: Range $30 to $100, Mean $66
- **MonthlyRate**: Range $2,094 to $26,999, Mean $14,313
- **PercentSalaryHike**: Range 11% to 25%, Mean 15.2%, Median 14%
- **StockOptionLevel**: Values 0, 1, 2, 3 (0 is the most common)

#### Job Information

- **Department**: Research and Development (65.4%), Sales (30.3%), Human Resources (4.3%)
- **JobRole**: 9 unique roles — Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, Human Resources
- **JobLevel**: Scale 1-5 (Levels 1 and 2 are most common, indicating a junior-heavy workforce)
- **JobInvolvement**: Scale 1-4 (Level 3 is most common)
- **BusinessTravel**: Travel_Rarely (71%), Travel_Frequently (19%), Non-Travel (10%)

#### Experience and Tenure

- **TotalWorkingYears**: Range 0 to 40, Mean 11.3 years
- **YearsAtCompany**: Range 0 to 40, Mean 7.0 years, Median 5 years
- **YearsInCurrentRole**: Range 0 to 18, Mean 4.2 years, Median 3 years
- **YearsSinceLastPromotion**: Range 0 to 15, Mean 2.2 years
- **YearsWithCurrManager**: Range 0 to 17, Mean 4.1 years
- **NumCompaniesWorked**: Range 0 to 9, Mean 2.7

#### Satisfaction and Well-being

- **JobSatisfaction**: Scale 1-4 (1=Low, 2=Medium, 3=High, 4=Very High)
- **EnvironmentSatisfaction**: Scale 1-4
- **RelationshipSatisfaction**: Scale 1-4
- **WorkLifeBalance**: Scale 1-4 (1=Bad, 2=Good, 3=Better, 4=Best)

#### Work Conditions

- **OverTime**: Yes (28.8%), No (71.2%)
- **DistanceFromHome**: Range 1 to 29 miles, Mean 9.2 miles
- **TrainingTimesLastYear**: Range 0 to 6, Mean 2.8

---

## 3. Exploratory Data Analysis Insights

### Attrition by Demographics

- **Age**: Employees aged 18-25 show the highest attrition rate at approximately 35%. Attrition decreases steadily with age. Employees above 45 show very low attrition below 10%.
- **Gender**: Male attrition rate (17%) is slightly higher than female (15%), but the difference is marginal.
- **Marital Status**: Single employees show the highest attrition rate (25%), compared to Married (13%) and Divorced (10%).

### Attrition by Compensation

- **Monthly Income**: Employees earning below $3,000 per month show attrition rates exceeding 25%. Employees earning above $8,000 per month show attrition rates below 10%. Income is one of the strongest predictors.
- **Stock Options**: Employees with Stock Option Level 0 (no options) have the highest attrition (25%). Levels 1-3 have significantly lower attrition (under 12%). This is the #1 most important feature in the model.
- **Salary Hike**: Higher salary hikes correlate with lower attrition, but the effect is moderate compared to base income.

### Attrition by Job Factors

- **Overtime**: Overtime workers have a 30% attrition rate, vs 10% for those who don't. This is a massive risk factor.
- **Department**: Sales has the highest attrition (21%), followed by HR (19%) and R&D (14%).
- **Job Role**: Sales Representatives (40%), Laboratory Technicians (24%), and HR (23%) have the highest turnover. Research Directors (2.5%) have the lowest.
- **Job Level**: Level 1 (Entry level) has the highest turnover (26%), dropping to below 7% for levels 3-5.
- **Job Satisfaction**: Low satisfaction (Level 1) drives 23% attrition, while Very High (Level 4) drops it to 11%.
- **Work-Life Balance**: Poor balance (Level 1) results in 31% attrition.

---

## 4. Machine Learning Model

- **Algorithm**: Gradient Boosting Classifier
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling Technique)
- **Accuracy**: 86%
- **ROC-AUC**: 0.80
- **Recall (Yes)**: 41%
- **Precision (Yes)**: 47%
- **F1 Score**: 0.44

*Interpretation: The model effectively identifies 4 out of 10 employees who actually leave, with 50% precision. The ROC-AUC of 0.80 indicates strong discriminative power.*

---

## 5. Feature Importance (Top 10)

1. **StockOptionLevel** — Most influential. Level 0 = High Risk.
2. **MonthlyIncome** — Danger zone: Below $3,000.
3. **OverTime** — Triples attrition risk.
4. **WorkLifeBalance** — Bad balance = 31% attrition.
5. **JobSatisfaction** — Low satisfaction = 23% attrition.
6. **Age** — Youth (18-25) = High Risk.
7. **YearsAtCompany** — High risk in first 2 years.
8. **TotalWorkingYears** — Less experience correlates with higher risk.
9. **JobLevel** — Entry level = 26% attrition.
10. **DistanceFromHome** — Long commutes increase risk.

---

## 6. Business Recommendations

- **Stock Options**: Extend even Level 1 options to entry-level and sales roles to stabilize workforce.
- **Overtime Control**: Employees working overtime are 3x more likely to leave. Monitor and reduce excessive hours.
- **Compensation Review**: Target raises for the bottom quartile (below $3k/mo) to yield high ROI on retention.
- **Sales Retainment**: Specific focus needed for Sales Representatives (40% attrition).
- **First 24 Months**: critical period; focus onboarding and mentoring on early-tenure employees.
- **Predictive Intervention**: Use the model output to conduct "stay interviews" with high-risk individuals.

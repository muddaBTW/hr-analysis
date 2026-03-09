import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ─── Page Config ───
st.set_page_config(layout='wide', page_title='HR Attrition Analysis')
st.title('📊 HR Attrition Analysis Dashboard')

# ─── Modern Color Palette ───
NO_COLOR = '#6366F1'
YES_COLOR = '#F43F5E'
ATTRITION_MAP = {'No': NO_COLOR, 'Yes': YES_COLOR}
CATEGORY_COLORS = [
    '#6366F1', '#F43F5E', '#14B8A6', '#F59E0B', '#8B5CF6',
    '#3B82F6', '#10B981', '#F97316', '#06B6D4'
]

# Matplotlib / Seaborn styling
sns.set_theme(style='whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFBFC',
    'axes.edgecolor': '#E5E7EB',
    'axes.labelcolor': '#374151',
    'axes.labelsize': 14,
    'text.color': '#1F2937',
    'xtick.color': '#6B7280',
    'ytick.color': '#6B7280',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'grid.color': '#F3F4F6',
    'font.size': 12,
})


def style_plotly(fig, title='', height=500):
    """Apply consistent styling to all plotly figures."""
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=17, color='#1F2937')),
        template='plotly_white',
        height=height,
        font=dict(family='sans-serif', size=14, color='#374151'),
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='white',
        margin=dict(t=60, b=50, l=60, r=30),
        xaxis=dict(tickfont=dict(size=13), title_font=dict(size=15)),
        yaxis=dict(tickfont=dict(size=13), title_font=dict(size=15)),
    )
    return fig


# ─── Load Data ───
df = pd.read_csv(
    r'C:\Users\mudda\OneDrive\Desktop\hr-ai-system\WA_Fn-UseC_-HR-Employee-Attrition.csv'
)

# ─── KPIs ───
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", "1,470")
col2.metric("Attrition Rate", "16.1%")
col3.metric("Avg Age", "37 yrs")
col4.metric("Avg Monthly Income", "$6,503")

st.divider()

# ─── Tabs ───
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📋 Feature Analysis",
    "🏢 Dept & Roles",
    "⚖️ Employee Insights",
    "🔗 Correlations",
    "🎯 Model Performance"
])


# ═══════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Attrition Distribution")
        counts = df['Attrition'].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_map=ATTRITION_MAP,
            hole=0.45,
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label+value',
            textfont_size=14,
            marker=dict(line=dict(color='white', width=2))
        )
        style_plotly(fig, 'Employee Attrition Split', 420)
        fig.update_layout(
            legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Key Insights")
        st.markdown("""
        - **237** employees left the company (**16.1%**)
        - **1,233** employees stayed (**83.9%**)
        - The dataset is **class-imbalanced** — SMOTE was applied during model training

        ---

        **Top 5 Attrition Drivers:**
        1. 🏷️ **Stock Option Level** — No options → 25%+ attrition
        2. 💰 **Monthly Income** — Below $3K → highest risk
        3. ⏰ **Overtime** — 3× higher attrition
        4. ⚖️ **Work-Life Balance** — Poor balance → 31% attrition
        5. 😊 **Job Satisfaction** — Low satisfaction → 23% attrition
        """)

    st.divider()

    if st.checkbox('Show Raw Dataset'):
        st.dataframe(df.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════
# TAB 2 — FEATURE ANALYSIS
# ═══════════════════════════════════════════════════
with tab2:
    st.subheader("Categorical Features vs Attrition")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols[:9]):
        sns.countplot(
            data=df, x=col, hue='Attrition',
            hue_order=['No', 'Yes'],
            palette=[NO_COLOR, YES_COLOR],
            ax=axes[i]
        )
        axes[i].set_title(col, fontsize=13, fontweight='bold', color='#1F2937')
        axes[i].tick_params(axis='x', rotation=30, labelsize=9)
        axes[i].set_xlabel('')

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

    st.info(
        "**Takeaway:** Overtime, business travel frequency, and department "
        "are the strongest categorical predictors. Sales roles and overtime "
        "workers show significantly elevated attrition."
    )

    st.divider()

    # ─── Numerical ───
    st.subheader("Numerical Features vs Attrition")

    num_cols = [
        'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'TotalWorkingYears'
    ]

    fig2, axes2 = plt.subplots(3, 3, figsize=(22, 16))
    axes2 = axes2.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(
            data=df, x='Attrition', y=col, hue='Attrition',
            order=['No', 'Yes'], hue_order=['No', 'Yes'],
            palette=ATTRITION_MAP, dodge=False, legend=False,
            ax=axes2[i]
        )
        axes2[i].set_title(col, fontsize=13, fontweight='bold', color='#1F2937')
        axes2[i].set_xlabel('')

    plt.tight_layout(pad=3.0)
    st.pyplot(fig2)

    st.info(
        "**Takeaway:** Employees who leave have lower monthly income, fewer "
        "total working years, and shorter tenure. Compensation and experience "
        "are the major quantitative attrition drivers."
    )


# ═══════════════════════════════════════════════════
# TAB 3 — DEPARTMENT & ROLES
# ═══════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)

    # ─── By Department ───
    with c1:
        st.subheader("Attrition Rate by Department")
        dept_rate = (
            df.groupby('Department')['Attrition']
            .value_counts(normalize=True)
            .mul(100).rename('Rate').reset_index()
        )
        dept_yes = dept_rate[dept_rate['Attrition'] == 'Yes']

        fig = px.bar(
            dept_yes, x='Department', y='Rate',
            color='Department',
            color_discrete_sequence=CATEGORY_COLORS,
            text=dept_yes['Rate'].round(1),
        )
        fig.update_traces(textposition='outside', texttemplate='%{text}%')
        style_plotly(fig, 'Attrition Rate (%) by Department', 450)
        fig.update_layout(showlegend=False, yaxis_title='Attrition Rate (%)', xaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        - **Sales** has the highest attrition (~21%)
        - **R&D** is lowest (~14%) despite being the largest dept
        """)

    # ─── By Job Role ───
    with c2:
        st.subheader("Attrition Rate by Job Role")
        role_rate = (
            df.groupby('JobRole')['Attrition']
            .value_counts(normalize=True)
            .mul(100).rename('Rate').reset_index()
        )
        role_yes = role_rate[role_rate['Attrition'] == 'Yes'].sort_values('Rate', ascending=True)

        fig = px.bar(
            role_yes, x='Rate', y='JobRole',
            orientation='h',
            color='Rate',
            color_continuous_scale=['#6366F1', '#F43F5E'],
            text=role_yes['Rate'].round(1),
        )
        fig.update_traces(textposition='outside', texttemplate='%{text}%')
        style_plotly(fig, 'Attrition Rate (%) by Job Role', 450)
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            xaxis_title='Attrition Rate (%)', yaxis_title=''
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        - **Sales Reps** have ~40% attrition (highest)
        - **Research Directors** have ~2.5% (lowest)
        """)


# ═══════════════════════════════════════════════════
# TAB 4 — EMPLOYEE INSIGHTS
# ═══════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)

    # ─── Work-Life Balance ───
    with c1:
        st.subheader("Work-Life Balance vs Attrition")
        wlb = (
            df.groupby('WorkLifeBalance')['Attrition']
            .value_counts(normalize=True).mul(100)
            .rename('Rate').reset_index()
        )
        wlb_yes = wlb[wlb['Attrition'] == 'Yes']

        fig = px.bar(
            wlb_yes, x='WorkLifeBalance', y='Rate',
            color='WorkLifeBalance',
            color_discrete_sequence=CATEGORY_COLORS,
            text=wlb_yes['Rate'].round(1),
        )
        fig.update_traces(textposition='outside', texttemplate='%{text}%')
        style_plotly(fig, 'Attrition Rate by Work-Life Balance', 400)
        fig.update_layout(showlegend=False, yaxis_title='Attrition Rate (%)', xaxis_title='Balance Level')
        st.plotly_chart(fig, use_container_width=True)

    # ─── Job Satisfaction ───
    with c2:
        st.subheader("Job Satisfaction vs Attrition")
        js = (
            df.groupby('JobSatisfaction')['Attrition']
            .value_counts(normalize=True).mul(100)
            .rename('Rate').reset_index()
        )
        js_yes = js[js['Attrition'] == 'Yes']

        fig = px.bar(
            js_yes, x='JobSatisfaction', y='Rate',
            color='JobSatisfaction',
            color_discrete_sequence=CATEGORY_COLORS,
            text=js_yes['Rate'].round(1),
        )
        fig.update_traces(textposition='outside', texttemplate='%{text}%')
        style_plotly(fig, 'Attrition Rate by Job Satisfaction', 400)
        fig.update_layout(showlegend=False, yaxis_title='Attrition Rate (%)', xaxis_title='Satisfaction Level')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    c3, c4 = st.columns(2)

    # ─── Income vs Years at Company ───
    with c3:
        st.subheader("Income vs Years at Company")
        fig = px.scatter(
            df, x='YearsAtCompany', y='MonthlyIncome',
            color='Attrition',
            color_discrete_map=ATTRITION_MAP,
            opacity=0.6,
        )
        fig.update_traces(marker=dict(size=5))
        style_plotly(fig, 'Monthly Income vs Tenure (by Attrition)', 450)
        st.plotly_chart(fig, use_container_width=True)

    # ─── Income Violin ───
    with c4:
        st.subheader("Income Distribution")
        fig = px.violin(
            df, x='Attrition', y='MonthlyIncome',
            color='Attrition',
            color_discrete_map=ATTRITION_MAP,
            box=True, points='all',
        )
        style_plotly(fig, 'Monthly Income by Attrition Status', 450)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Takeaway:** Poor work-life balance causes ~31% attrition. Low satisfaction "
        "drives ~23%. Employees who leave cluster in lower income and shorter tenure ranges."
    )


# ═══════════════════════════════════════════════════
# TAB 5 — CORRELATIONS
# ═══════════════════════════════════════════════════
with tab5:
    st.subheader("Feature Correlation Heatmap")
    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        color_continuous_scale=['#6366F1', '#FAFBFC', '#F43F5E'],
        aspect='auto',
        text_auto='.2f',
    )
    style_plotly(fig, 'Numerical Feature Correlations', 1000)
    fig.update_layout(
        width=1300,
        xaxis=dict(tickfont=dict(size=12), tickangle=-45),
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=150, b=150, t=60, r=30),
    )
    fig.update_traces(textfont_size=8)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Key correlations:** MonthlyIncome ↔ JobLevel (0.95), "
        "MonthlyIncome ↔ TotalWorkingYears (0.77), "
        "Age ↔ TotalWorkingYears (0.68). "
        "Income, experience, and seniority are structurally aligned."
    )


# ═══════════════════════════════════════════════════
# TAB 6 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════
with tab6:
    model = joblib.load("../backend/model.pkl")
    columns = joblib.load("../backend/columns.pkl")

    X = pd.get_dummies(df.drop('Attrition', axis=1))
    X = X.reindex(columns=columns, fill_value=0)
    y = df['Attrition'].map({'No': 0, 'Yes': 1})
    y_prob = model.predict_proba(X)[:, 1]

    c1, c2 = st.columns(2)

    # ─── ROC ───
    with c1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            line=dict(color=NO_COLOR, width=2.5),
            name=f'AUC = {roc_auc:.3f}', fill='tozeroy',
            fillcolor='rgba(99,102,241,0.15)'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(color='#D1D5DB', dash='dash', width=1),
            name='Random', showlegend=False
        ))
        style_plotly(fig, f'ROC Curve  (AUC = {roc_auc:.3f})', 420)
        fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

    # ─── Precision-Recall ───
    with c2:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y, y_prob)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode='lines',
            line=dict(color=YES_COLOR, width=2.5),
            name='PR Curve', fill='tozeroy',
            fillcolor='rgba(244,63,94,0.15)'
        ))
        style_plotly(fig, 'Precision–Recall Curve', 420)
        fig.update_layout(xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─── Feature Importance (aggregated by original feature) ───
    st.subheader("Top 15 Feature Importance")
    importance = model.feature_importances_
    original_cols = df.drop('Attrition', axis=1).columns.tolist()

    # Map each one-hot column back to its original feature and sum importances
    aggregated = {}
    for col_name, imp in zip(X.columns, importance):
        matched = col_name  # default: keep as-is for numeric features
        for orig in original_cols:
            if col_name == orig or col_name.startswith(orig + '_'):
                matched = orig
                break
        aggregated[matched] = aggregated.get(matched, 0) + imp

    # Sort and take top 15
    sorted_feats = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:15]
    feat_names = [f[0] for f in sorted_feats][::-1]  # reverse for horizontal bar
    feat_values = [f[1] for f in sorted_feats][::-1]

    fig = px.bar(
        x=feat_values, y=feat_names, orientation='h',
        color=feat_values,
        color_continuous_scale=['#6366F1', '#F43F5E'],
        labels={'x': 'Importance', 'y': 'Feature'},
    )
    style_plotly(fig, 'Top 15 Predictive Features (Aggregated)', 500)
    fig.update_layout(coloraxis_showscale=False, yaxis_title='', xaxis_title='Feature Importance')
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Key drivers:** OverTime, income-related variables, and tenure features "
        "dominate predictive importance. Compensation, workload, and experience "
        "are the primary levers for HR retention strategies."
    )
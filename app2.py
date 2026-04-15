import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px

# ================= CONFIG =================
st.set_page_config(layout="wide")

# ================= UI (GLASS) =================
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
    transition: 0.3s;
}

.glass:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
}

.metric-title {
    font-size: 14px;
    color: #94a3b8;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #f8fafc;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "login" not in st.session_state:
    st.session_state.login = False
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ================= LOGIN =================
def login():
    st.title("🏦 Smart Churn Prediction System")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "1234":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid credentials")

def logout():
    st.session_state.login = False

# ================= LOAD =================
@st.cache_data
def load():
    df = pd.read_csv("data/Churn_Modelling.csv")
    fb = pd.read_csv("data/customer_feedback.csv")
    return df, fb

df, fb = load()

# ================= MODEL =================

MODEL_PATH = "models/xgb_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Features used during training
EXPECTED_FEATURES = scaler.feature_names_in_

# ================= PREPROCESS =================
df_model = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])

# Convert categorical → numerical
df_model = pd.get_dummies(df_model, columns=["Gender", "Geography"], drop_first=True)

# Align with training features
df_model = df_model.reindex(columns=EXPECTED_FEATURES, fill_value=0)

# Scale input
scaled_data = scaler.transform(df_model)

# ================= PREDICTION =================
df["Probability"] = np.round(model.predict_proba(scaled_data)[:, 1] * 100, 2)

# ================= RISK =================
def risk(x):
    if x>=85: return "Critical"
    elif x>=65: return "High"
    elif x>=45: return "Medium"
    else: return "Low"

df["Risk"] = df["Probability"].apply(risk)

# ================= FEEDBACK =================
fb.columns = fb.columns.str.strip()

issue_col = "Issue_Type" if "Issue_Type" in fb.columns else fb.columns[1]
severity_col = "Issue Level" if "Issue Level" in fb.columns else fb.columns[2]

fb_summary = fb.groupby("CustomerId").agg(
    Complaint_Count=("CustomerId","count"),
    Issue=(issue_col,"first"),
    High_Issue=(severity_col, lambda x:(x=="High").sum())
).reset_index()

df = df.merge(fb_summary, on="CustomerId", how="left")
df.fillna({"Complaint_Count":0,"Issue":" - ","High_Issue":0}, inplace=True)

# ================= EXPLANATION =================
def explain(r):

    reasons = []

    if r["IsActiveMember"] == 0:
        reasons.append("Customer is not actively using banking services, indicating low engagement behavior")

    if r["NumOfProducts"] == 1:
        reasons.append("Customer is using only one banking product, showing weak relationship with bank")

    if r["Balance"] > 150000:
        reasons.append("Customer holds high account balance but may shift to competitors")

    if r["Tenure"] < 3:
        reasons.append("Customer has short relationship period with the bank")

    if r["CreditScore"] < 500:
        reasons.append("Customer has low credit score indicating financial risk")

    if r["Complaint_Count"] >= 2:
        reasons.append("Customer has raised multiple complaints, showing dissatisfaction")

    if r["High_Issue"] >= 1:
        reasons.append("Customer has reported high severity issues affecting trust")

    if not reasons:
        return "Customer profile is stable with no strong churn indicators"

    return " | ".join(reasons)
# ================= RECOMMENDATION =================
def recommend(r):

    actions = []

    if r["Risk"] == "Critical":
        actions.append("Immediate retention call by relationship manager")
        actions.append("Provide personalized retention offer")
        actions.append("Resolve all pending issues urgently")

    if r["Risk"] == "High":
        actions.append("Initiate proactive engagement with customer")
        actions.append("Offer targeted financial products or benefits")

    if r["Complaint_Count"] >= 2:
        actions.append("Prioritize complaint resolution and follow-up")

    if r["IsActiveMember"] == 0:
        actions.append("Re-engage customer through personalized campaigns")

    if r["NumOfProducts"] == 1:
        actions.append("Cross-sell additional banking services")

    if r["Balance"] > 150000:
        actions.append("Assign dedicated relationship manager for high-value customer")

    if not actions:
        return "Continue regular monitoring and maintain engagement"

    return " | ".join(actions)

#========================================
# APPLY LOGIC
df["Explanation"] = df.apply(explain, axis=1)
df["Recommendation"] = df.apply(recommend, axis=1)

# ================= NAV =================
def nav():

    # ===================== HEADER =====================
    st.sidebar.markdown("""
    <h2 style='color:#38bdf8; text-align:center; margin-bottom:5px;'>
     Smart Churn Prediction
    </h2>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <p style='color:#94a3b8; text-align:center; font-size:12px; margin-bottom:20px;'>
    Predict • Analyze • Retain
    </p>
    """, unsafe_allow_html=True)

    # ===================== INIT PAGE =====================
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    # ===================== BUTTON STYLE =====================
    st.sidebar.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 8px;
        background: linear-gradient(135deg,#020617,#0f172a);
        color: #e2e8f0;
        border: 1px solid #1e293b;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg,#1e293b,#020617);
        border: 1px solid #38bdf8;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

    # ===================== NAV BUTTONS =====================
    if st.sidebar.button("Dashboard"):
        st.session_state.page = "Dashboard"

    if st.sidebar.button("Feedback"):
        st.session_state.page = "Feedback"

    if st.sidebar.button("Deep Analysis"):
        st.session_state.page = "Deep"

    if st.sidebar.button("Churn Drivers"):
        st.session_state.page = "Drivers"

    if st.sidebar.button("Testing"):
        st.session_state.page = "Testing"

    if st.sidebar.button("Report"):
        st.session_state.page = "Report"

    # ===================== LOGOUT =====================
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    if st.sidebar.button(" Logout"):
        st.session_state.login = False
        st.rerun()

    return st.session_state.page
# ================= DASHBOARD =================
def dashboard():

    st.title("📊 Customer Dashboard")

    # ===================== DATA =====================
    critical = len(df[df["Risk"] == "Critical"])
    high = len(df[df["Risk"] == "High"])
    medium = len(df[df["Risk"] == "Medium"])
    low = len(df[df["Risk"] == "Low"])

    # ===================== CSS =====================
    st.markdown("""
    <style>
    .card {
        background: linear-gradient(135deg,#020617,#0f172a);
        padding:20px;
        border-radius:14px;
        text-align:center;
        border:1px solid rgba(59,130,246,0.2);
        box-shadow: 0 6px 25px rgba(0,0,0,0.7);
        transition: 0.3s;
    }
    .card:hover {
        transform: translateY(-6px);
        box-shadow: 0 10px 35px rgba(0,0,0,0.9);
    }
    .title {
        font-size:14px;
        color:#94a3b8;
    }
    .critical {
        font-size:34px;
        font-weight:700;
        color:#fb7185;
        text-shadow:0 0 12px rgba(251,113,133,0.7);
    }
    .high {
        font-size:34px;
        font-weight:700;
        color:#facc15;
        text-shadow:0 0 12px rgba(250,204,21,0.7);
    }
    .medium {
        font-size:34px;
        font-weight:700;
        color:#38bdf8;
        text-shadow:0 0 12px rgba(56,189,248,0.7);
    }
    .low {
        font-size:34px;
        font-weight:700;
        color:#4ade80;
        text-shadow:0 0 12px rgba(74,222,128,0.7);
    }
    .section {
        background: linear-gradient(135deg,#020617,#0f172a);
        padding:18px;
        border-radius:12px;
        border:1px solid #1e293b;
        box-shadow: 0 4px 20px rgba(0,0,0,0.7);
        margin-top:10px;
        color:#e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===================== CARDS =====================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="title">Critical Risk</div>
            <div class="critical">{critical}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="title">High Risk</div>
            <div class="high">{high}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="title">Medium Risk</div>
            <div class="medium">{medium}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="title">Low Risk</div>
            <div class="low">{low}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===================== BUSINESS INSIGHTS =====================
    st.markdown("### 🔥 Business Insights")

    st.markdown("""
    <div class="section">
    • Single product customers show higher churn <br>
    • Inactive customers are major risk <br>
    • High balance customers need attention <br>
    • Complaints increase churn probability
    </div>
    """, unsafe_allow_html=True)

    # ===================== PRIORITY ACTION =====================
    st.markdown("### ⚡ Priority Action")

    st.markdown(f"""
    <div class="section">
    Focus on <b style="color:#fb7185;">{critical}</b> critical customers immediately.<br><br>
    Assign managers and resolve issues to prevent churn.
    </div>
    """, unsafe_allow_html=True)

# ================= FEEDBACK =================
def feedback_page():

    st.title("💬 Customer Feedback Intelligence")

    import plotly.express as px

    # ===================== DATA =====================
    total_feedback = len(fb)

    # repeated complaints (same customer multiple times)
    repeated = fb["CustomerId"].value_counts()
    repeated = len(repeated[repeated > 1])

    # ===================== CSS =====================
    st.markdown("""
    <style>
    .stat-card {
        background: linear-gradient(135deg,#020617,#0f172a);
        padding:18px;
        border-radius:14px;
        border:1px solid rgba(59,130,246,0.25);
        box-shadow: 0 6px 25px rgba(0,0,0,0.7);
        transition: 0.3s;
        text-align:center;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 35px rgba(0,0,0,0.9);
    }
    .stat-title {
        color:#94a3b8;
        font-size:14px;
    }
    .stat-value {
        font-size:32px;
        font-weight:700;
        color:#38bdf8;
        text-shadow:0 0 12px rgba(56,189,248,0.7);
    }
    .stat-alert {
        font-size:32px;
        font-weight:700;
        color:#f87171;
        text-shadow:0 0 12px rgba(248,113,113,0.7);
    }
    .section-box {
        background: linear-gradient(135deg,#020617,#0f172a);
        padding:16px;
        border-radius:12px;
        border:1px solid #1e293b;
        box-shadow: 0 4px 20px rgba(0,0,0,0.7);
        color:#e2e8f0;
        margin-top:10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===================== KPI CARDS =====================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">📊 Total Feedbacks</div>
            <div class="stat-value">{total_feedback}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">⚠️ Repeated Complaints</div>
            <div class="stat-alert">{repeated}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===================== COMPLAINT DISTRIBUTION =====================
    st.markdown("### 📊 Complaint Distribution")

    # auto detect issue column
    issue_col = "Issue" if "Issue" in fb.columns else fb.columns[1]

    issue_counts = fb[issue_col].value_counts().reset_index()
    issue_counts.columns = ["Issue", "Count"]

    fig = px.bar(
        issue_counts,
        x="Issue",
        y="Count",
        color="Count",
        color_continuous_scale=["#0ea5e9", "#38bdf8", "#7dd3fc"]
    )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===================== INSIGHTS =====================
    st.markdown("### 💡 Insights")

    st.markdown("""
    <div class="section-box">
    • Customers with repeated complaints show higher churn risk <br>
    • Service-related issues directly impact retention <br>
    • Faster resolution improves customer satisfaction <br>
    • Complaint trends help identify weak service areas
    </div>
    """, unsafe_allow_html=True)

    # ===================== ACTION =====================
    st.markdown("### 🚀 Recommended Actions")

    st.markdown("""
    <div class="section-box">
    • Identify high-frequency complaint customers <br>
    • Improve response time for critical issues <br>
    • Provide compensation or retention offers <br>
    • Monitor complaint patterns continuously
    </div>
    """, unsafe_allow_html=True)
# ================= DEEP =================
def deep():

    st.title("🔍 Customer Deep Analysis")

    cid = st.text_input("Enter Customer ID")

    if cid:
        try:
            cid = int(cid)
            row = df[df["CustomerId"]==cid]

            if row.empty:
                st.error("Customer not found")
            else:
                r = row.iloc[0]

                col1,col2,col3 = st.columns(3)

                col1.metric("Churn Probability", f"{r['Probability']}%")
                col2.metric("Risk Level", r["Risk"])
                col3.metric("Complaint Count", int(r["Complaint_Count"]))

                st.markdown("### 📌 Customer Issues")
                st.write(r["Issue"] if r["Issue"] != " - " else "No major issues reported")

                st.markdown("### 🧠 Explanation")
                st.info(r["Explanation"])

                st.markdown("### 💡 Recommendation")
                st.success(r["Recommendation"])

        except:
            st.error("Invalid ID")

#===================Drivers================
def drivers():

    st.title("🧠 Churn Drivers & Feature Analysis")

    import pandas as pd
    import plotly.express as px
    import numpy as np

    # ===================== FEATURES =====================
    features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "EstimatedSalary",
        "HasCrCard",
        "EngagementScore",
        "FinancialStrength",
        "HighBalance_LowEngagement"
    ]

    geo_cols = [col for col in df.columns if "Geography_" in col]
    final_features = [f for f in features if f in df.columns] + geo_cols

    # ===================== IMPORTANCE =====================
    importance = df[final_features].corrwith(df["Exited"]).abs()

    imp_df = (
        pd.DataFrame({
            "Feature": importance.index,
            "Importance": importance.values
        })
        .sort_values(by="Importance", ascending=True)
        .tail(10)
    )

    # Clean names
    imp_df["Feature"] = imp_df["Feature"].str.replace("Geography_", "Geo: ")

    
    # ===================== BAR CHART =====================
    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=[
            "#0ea5e9", "#38bdf8", "#60a5fa", "#3b82f6"
        ]
    )

    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===================== TOP FEATURE HIGHLIGHT =====================
    top_feature = imp_df.iloc[-1]["Feature"]

    st.markdown(f"""
    <div style='
        background: linear-gradient(90deg,#0ea5e9,#1e40af);
        padding:12px;
        border-radius:10px;
        margin-top:10px;
        text-align:center;
        color:white;
        font-weight:600;
        box-shadow:0 3px 10px rgba(0,0,0,0.4);
    '>
         Most Influential Feature: {top_feature}
    </div>
    """, unsafe_allow_html=True)

    # ===================== OBSERVATIONS =====================
    st.markdown("### 💡 Key Observations")

    st.markdown("""
<div style='
    background:#020617;
    padding:15px;
    border-radius:10px;
    border:1px solid #1e293b;
    box-shadow:0 3px 10px rgba(0,0,0,0.5);
    color:#e2e8f0;
'>
• Customer activity plays a major role in churn <br>
• Number of products strongly affects retention <br>
• Balance and tenure influence customer stability <br>
• Low engagement significantly increases churn probability <br>
• High-value customers with low engagement are at highest risk
</div>
""", unsafe_allow_html=True)
 #=======================Testing=============================================   

def testing():

    st.title("Testing Customer Data")

    c1, c2 = st.columns(2)

    with c1:
        credit = st.number_input("Customer Credit Score", value=480)
        age = st.number_input("Age", value=55)
        tenure = st.number_input("Years with Bank", value=1)

        products = st.selectbox(
            "Number of Bank Services Used",
            [1,2,3,4], index=0
        )

    with c2:
        balance = st.number_input(
            "Current Account Balance",
            value=150000.0
        )

        salary = st.number_input(
            "Estimated Annual Income",
            value=30000.0
        )

        has_card = st.selectbox(
            "Customer Has Credit Card?",
            ["Yes","No"]
        )

        is_active = st.selectbox(
            "Customer Actively Using Services?",
            ["Yes","No"]
        )

    has_card = 1 if has_card == "Yes" else 0
    is_active = 1 if is_active == "Yes" else 0

    if st.button("🔍 Predict Churn", use_container_width=True):

        input_df = pd.DataFrame([{
            "CreditScore": credit,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary,
            "Geography_Germany": 0,
            "Geography_Spain": 0,
            "Gender_Male": 1
        }])

        input_df = input_df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

        scaled = scaler.transform(input_df)

        prob = model.predict_proba(scaled)[0][1] * 100

        risk_level = risk(prob)

        st.markdown("### 📊 Prediction Result")

        if risk_level == "Critical":
            st.error(f"🔴 CRITICAL RISK — {prob:.2f}%")

        elif risk_level == "High":
            st.warning(f"🟠 HIGH RISK — {prob:.2f}%")

        elif risk_level == "Medium":
            st.info(f"🟡 MEDIUM RISK — {prob:.2f}%")

        else:
            st.success(f"🟢 LOW RISK — {prob:.2f}%")

        # Explanation + Recommendation (reuse your logic)
        temp = input_df.iloc[0].to_dict()
        temp["Risk"] = risk_level
        temp["Complaint_Count"] = 0
        temp["High_Issue"] = 0

        st.markdown("### 🧠 Explanation")
        st.info(explain(temp))

        st.markdown("### 💡 Recommendation")
        st.success(recommend(temp))

# ================= REPORT =================
def report():

    st.title("📁 Business Retention Report")

    import plotly.graph_objects as go

    st.markdown("### 📊 Churn Analytics")

    churned = df["Exited"].sum()
    stayed = len(df) - churned

    fig = go.Figure(data=[go.Pie(
        labels=["Stayed", "Churned"],
        values=[stayed, churned],
        hole=0.65,
        marker=dict(
            colors=["#38bdf8", "#1e40af"],
            line=dict(color="#0f172a", width=2)
        ),
        textinfo="label+percent",
        textfont=dict(size=14, color="#e2e8f0"),
        hoverinfo="label+value+percent"
    )])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📥 Export Business Report")

    if st.button("Generate Report", key="report_btn"):

        rep = df[df["Risk"] != "Low"].copy()

        rep = rep.sort_values(by="Probability", ascending=False).reset_index(drop=True)

        required_cols = [
            "CustomerId",
            "Probability",
            "Risk",
            "Issue",
            "Complaint_Count",
            "Explanation",
            "Recommendation"
        ]

        for col in required_cols:
            if col not in rep.columns:
                rep[col] = " - "

        final = rep[required_cols]

        file = f"Report_{datetime.now().strftime('%Y%m%d')}.xlsx"

        final.to_excel(file, index=False)

        st.success("✅ Report Generated Successfully")

        with open(file, "rb") as f:
            st.download_button("⬇ Download Report", f, file_name=file)
# ================= MAIN =================
def app():
    nav()

    if st.session_state.page == "Dashboard":
        dashboard()
    elif st.session_state.page == "Feedback":
        feedback_page()
    elif st.session_state.page == "Deep":
        deep()
    elif st.session_state.page == "Drivers":
        drivers()
    elif st.session_state.page == "Testing":
        testing()
    elif st.session_state.page == "Report":
        report()

if not st.session_state.login:
    login()
else:
    app()
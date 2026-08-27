import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import datetime
import os
import json

from cli import init_db, connect_db, log_user_override
from categorizer_ml import predict
from retrain import run_retraining_pipeline

# Configure Streamlit Page
st.set_page_config(
    page_title="SpendSmart - ML Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism, Vibrant Dark Mode Accents)
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .prediction-badge {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .fallback-badge {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database Schema
init_db()

def load_expenses_df():
    """Loads expenses from SQLite database into pandas DataFrame."""
    conn = connect_db()
    query = """
        SELECT id, title, amount, category, date, description, 
               predicted_category, confidence_score, is_user_corrected 
        FROM expense ORDER BY date DESC, id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = df['amount'].astype(float)
    return df

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/96/money-bag-euro.png", width=70)
st.sidebar.title("SpendSmart ML")
st.sidebar.caption("Categorization Engine v1.0")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Analytics Dashboard",
        "➕ Add Expense (Live ML)",
        "📋 Expenses & Corrections Log",
        "🤖 Model Lifecycle & Retraining"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💱 Currency Preference")
currency_pref = st.sidebar.radio("Display Currency:", ["USD ($)", "INR (₹)"])
curr_symbol = "₹" if "INR" in currency_pref else "$"
curr_rate = 83.0 if "INR" in currency_pref else 1.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ ML Engine Status")
st.sidebar.success("Model Active: `model_v1.joblib` (LogReg / NaiveBayes)")

# ----------------------------------------------------
# PAGE 1: ANALYTICS DASHBOARD
# ----------------------------------------------------
if nav_choice == "📊 Analytics Dashboard":
    st.markdown('<div class="main-header">📊 Financial Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time spend breakdowns, category distributions, and monthly trends.</div>', unsafe_allow_html=True)

    df = load_expenses_df()

    if df.empty:
        st.info("No expense data found in `expenses.db`. Use '➕ Add Expense' to submit transactions or load sample data!")
    else:
        # Calculate display amount in selected currency
        df['disp_amount'] = df['amount'] * curr_rate

        # Top KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        total_spend = df['disp_amount'].sum()
        total_tx = len(df)
        corrected_cnt = df['is_user_corrected'].sum()
        ml_accuracy_rate = 100.0 * (1 - (corrected_cnt / total_tx)) if total_tx > 0 else 100.0

        col1.metric("💰 Total Spend", f"{curr_symbol}{total_spend:,.2f}")
        col2.metric("📝 Total Expenses", f"{total_tx}")
        col3.metric("🔄 User Corrections", f"{corrected_cnt}")
        col4.metric("🎯 Acceptance Rate", f"{ml_accuracy_rate:.1f}%")

        st.markdown("---")

        # Visual Charts Grid
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Category Spend Breakdown")
            cat_df = df.groupby('category')['disp_amount'].sum().reset_index()
            fig_pie = px.pie(
                cat_df, 
                values='disp_amount', 
                names='category',
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader(f"Spending by Category ({curr_symbol})")
            fig_bar = px.bar(
                cat_df,
                x='category',
                y='disp_amount',
                color='category',
                text_auto='.2f',
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title=f"Amount ({curr_symbol})")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Monthly Spending Trend Line
        st.subheader("📅 Monthly Spending Trend")
        df['year_month'] = df['date'].dt.to_period('M').astype(str)
        monthly_df = df.groupby('year_month')['disp_amount'].sum().reset_index()
        
        fig_line = px.line(
            monthly_df,
            x='year_month',
            y='disp_amount',
            markers=True,
            line_shape='spline',
            color_discrete_sequence=['#4F46E5']
        )
        fig_line.update_layout(xaxis_title="Month", yaxis_title=f"Total Amount ({curr_symbol})")
        st.plotly_chart(fig_line, use_container_width=True)

# ----------------------------------------------------
# PAGE 2: ADD EXPENSE (LIVE ML AUTO-CATEGORIZATION)
# ----------------------------------------------------
elif nav_choice == "➕ Add Expense (Live ML)":
    st.markdown('<div class="main-header">➕ Add Expense with Live ML Categorization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">As you type your expense title and description, ML Model v1 predicts the category in real-time. Enter amounts in USD ($) or INR (₹).</div>', unsafe_allow_html=True)

    with st.form("add_expense_form", clear_on_submit=True):
        f_col1, f_col2 = st.columns(2)

        with f_col1:
            title_input = st.text_input("Transaction Title *", placeholder="e.g. Swiggy Order, Uber Ride, Electricity Bill")
            input_currency = st.selectbox("Select Input Currency *", ["USD ($)", "INR (₹)"])
            symbol_in = "₹" if "INR" in input_currency else "$"
            amount_input = st.number_input(f"Amount ({symbol_in}) *", min_value=0.01, value=500.00 if "INR" in input_currency else 25.00, step=1.00)
            
            # Currency conversion logic
            amount_usd = amount_input / 83.0 if "INR" in input_currency else amount_input
            if "INR" in input_currency:
                st.caption(f"💱 Live Conversion: **₹{amount_input:,.2f} INR** ≈ **${amount_usd:,.2f} USD**")
            else:
                st.caption(f"💱 Live Conversion: **${amount_input:,.2f} USD** ≈ **₹{amount_input * 83.0:,.2f} INR**")

            date_input = st.date_input("Transaction Date", value=datetime.date.today())

        with f_col2:
            desc_input = st.text_area("Description (optional)", placeholder="e.g. Lunch with colleagues, monthly utility bill")

        # Live Real-time ML Prediction
        full_text = f"{title_input} {desc_input}".strip()
        
        if full_text:
            pred_cat, conf_score, source = predict(full_text, amount_usd)
        else:
            pred_cat, conf_score, source = ("Other", 0.10, "rule_fallback")

        st.markdown("### 🤖 ML Model Live Prediction")
        
        b_col1, b_col2 = st.columns([1, 2])
        with b_col1:
            if source == "ml_model":
                st.markdown(f'<div class="prediction-badge">🤖 ML Model v1: {pred_cat}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="fallback-badge">⚡ Rule Fallback: {pred_cat}</div>', unsafe_allow_html=True)
        
        with b_col2:
            st.progress(float(conf_score))
            st.caption(f"Prediction Confidence Score: **{conf_score:.2f}** ({source})")

        # Category Selector pre-selected with prediction
        categories_list = ["Food", "Transport", "Entertainment", "Utilities", "Shopping", "Other"]
        default_index = categories_list.index(pred_cat) if pred_cat in categories_list else 5

        selected_category = st.selectbox(
            "Confirm or Override Category:",
            options=categories_list,
            index=default_index
        )

        submit_btn = st.form_submit_button("🚀 Submit Expense", use_container_width=True)

        if submit_btn:
            if not title_input:
                st.error("Please enter a transaction title.")
            else:
                is_corrected = 1 if selected_category.lower() != pred_cat.lower() else 0
                date_str = date_input.strftime("%Y-%m-%d")

                conn = connect_db()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO expense (title, amount, category, date, description, predicted_category, confidence_score, is_user_corrected)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (title_input, amount_usd, selected_category, date_str, desc_input, pred_cat, conf_score, is_corrected))
                conn.commit()
                conn.close()

                if is_corrected == 1:
                    log_user_override(title_input, amount_usd, selected_category, date_str, desc_input, pred_cat, conf_score)
                    st.warning(f"Expense added ({symbol_in}{amount_input:,.2f})! User Override logged: '{pred_cat}' ➔ '{selected_category}'.")
                else:
                    st.success(f"Expense added successfully ({symbol_in}{amount_input:,.2f})! Category: '{selected_category}'")

# ----------------------------------------------------
# PAGE 3: EXPENSE RECORDS & CORRECTIONS LOG
# ----------------------------------------------------
elif nav_choice == "📋 Expenses & Corrections Log":
    st.markdown('<div class="main-header">📋 Expense Records & Feedback Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">View full transaction history, ML predictions, confidence scores, and user overrides.</div>', unsafe_allow_html=True)

    df = load_expenses_df()

    if df.empty:
        st.info("No expense entries found.")
    else:
        df['disp_amount'] = df['amount'] * curr_rate

        # Filters
        f1, f2 = st.columns(2)
        with f1:
            filter_cat = st.multiselect("Filter by Category:", options=df['category'].unique(), default=df['category'].unique())
        with f2:
            only_corrected = st.checkbox("Show Only User Corrected Transactions", value=False)

        filtered_df = df[df['category'].isin(filter_cat)]
        if only_corrected:
            filtered_df = filtered_df[filtered_df['is_user_corrected'] == 1]

        st.dataframe(
            filtered_df[['id', 'date', 'title', 'disp_amount', 'category', 'predicted_category', 'confidence_score', 'is_user_corrected', 'description']],
            use_container_width=True,
            column_config={
                "id": "ID",
                "disp_amount": st.column_config.NumberColumn(f"Amount ({curr_symbol})", format=f"{curr_symbol}%.2f"),
                "confidence_score": st.column_config.NumberColumn("Conf Score", format="%.2f"),
                "is_user_corrected": st.column_config.CheckboxColumn("User Corrected?")
            }
        )

        st.markdown("---")
        st.subheader("🔄 Active User Override Feedback CSV (`user_feedback_data.csv`)")
        if os.path.exists("user_feedback_data.csv"):
            fb_df = pd.read_csv("user_feedback_data.csv")
            st.dataframe(fb_df, use_container_width=True)
        else:
            st.info("No user overrides logged yet in `user_feedback_data.csv`.")

# ----------------------------------------------------
# PAGE 4: MODEL LIFECYCLE & RETRAINING
# ----------------------------------------------------
elif nav_choice == "🤖 Model Lifecycle & Retraining":
    st.markdown('<div class="main-header">🤖 ML Model Lifecycle & Retraining</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Track model metrics progression, confusion matrices, and trigger continuous retraining.</div>', unsafe_allow_html=True)

    retrain_col1, retrain_col2 = st.columns([2, 1])

    with retrain_col1:
        st.subheader("🔄 Continuous ML Retraining")
        st.write("Retrain the ML classification model using all original data plus newly logged user overrides.")

    with retrain_col2:
        if st.button("🚀 Trigger Model Retraining", type="primary", use_container_width=True):
            with st.spinner("Retraining model on updated corpus..."):
                entry = run_retraining_pipeline()
                st.success(f"Model retrained! New Accuracy: {entry['accuracy']:.2f}% (Dataset size: {entry['total_samples']})")

    st.markdown("---")

    # Metrics History
    st.subheader("📈 ML Lifecycle Accuracy Progression")
    if os.path.exists("training_history.json"):
        with open("training_history.json", "r") as f:
            hist_data = json.load(f)
        hist_df = pd.DataFrame(hist_data)
        st.dataframe(hist_df, use_container_width=True)

    # Render Plot Images if available
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.subheader("Accuracy Growth Trend")
        if os.path.exists("accuracy_over_time.png"):
            st.image("accuracy_over_time.png", use_container_width=True)
        else:
            st.info("No accuracy trend chart available.")
    
    with v_col2:
        st.subheader("Confusion Matrix Grid")
        if os.path.exists("confusion_matrices.png"):
            st.image("confusion_matrices.png", use_container_width=True)
        else:
            st.info("No confusion matrix chart available.")
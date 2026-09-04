
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Consumer Banking | Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATA + MODEL
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "credit_risk_processed.csv"
MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(" Could not load the project data/model.")
    st.exception(e)
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.title(" Consumer Banking — Credit Risk Analytics")
st.caption(
    "Credit Risk Pipeline | Portfolio Analysis | Borrower Risk Prediction"
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(" Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Executive Overview",
        "Risk Explorer",
        "Credit Risk Predictor",
        "Dataset Preview"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    **Project:** Consumer Banking Credit Risk Pipeline

    **Model:** Random Forest Classifier

    **Purpose:** Identify borrower credit-risk patterns and estimate default probability.
    """
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_default_rate(data):
    if "loan_status" not in data.columns or len(data) == 0:
        return 0

    return data["loan_status"].mean() * 100


def money(value):
    return f"${value:,.0f}"


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header(" Executive Overview")

    # ----------------------------
    # KPI calculations
    # ----------------------------

    total_borrowers = len(df)

    default_rate = get_default_rate(df)

    if "loan_amnt" in df.columns:
        avg_loan = df["loan_amnt"].mean()
    else:
        avg_loan = 0

    if "person_income" in df.columns:
        avg_income = df["person_income"].mean()
    else:
        avg_income = 0

    # ----------------------------
    # KPI cards
    # ----------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            " Total Borrowers",
            f"{total_borrowers:,}"
        )

    with col2:
        st.metric(
            " Default Rate",
            f"{default_rate:.2f}%"
        )

    with col3:
        st.metric(
            " Average Loan",
            money(avg_loan)
        )

    with col4:
        st.metric(
            " Average Income",
            money(avg_income)
        )

    st.divider()

    # ========================================================
    # DEFAULT RATE BY LOAN GRADE
    # ========================================================

    st.subheader(" Default Rate by Loan Grade")

    if "loan_grade" in df.columns:

        grade_analysis = (
            df.groupby("loan_grade")["loan_status"]
            .mean()
            .mul(100)
            .reset_index()
        )

        grade_analysis.columns = [
            "Loan Grade",
            "Default Rate (%)"
        ]

        grade_analysis = grade_analysis.sort_values(
            "Default Rate (%)",
            ascending=False
        )

        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.bar_chart(
                grade_analysis.set_index("Loan Grade")
            )

        with col2:
            st.dataframe(
                grade_analysis,
                use_container_width=True,
                hide_index=True
            )

    # ========================================================
    # DEFAULT RATE BY LOAN INTENT
    # ========================================================

    st.subheader(" Default Rate by Loan Intent")

    if "loan_intent" in df.columns:

        intent_analysis = (
            df.groupby("loan_intent")["loan_status"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        st.bar_chart(intent_analysis)

    # ========================================================
    # HOME OWNERSHIP
    # ========================================================

    st.subheader(" Borrowers by Home Ownership")

    if "person_home_ownership" in df.columns:

        home_counts = (
            df["person_home_ownership"]
            .value_counts()
        )

        st.bar_chart(home_counts)


# ============================================================
# RISK EXPLORER
# ============================================================

elif page == "Risk Explorer":

    st.header("🔎 Credit Risk Explorer")

    st.write(
        "Explore borrower default patterns using different portfolio segments."
    )

    # ========================================================
    # FILTERS
    # ========================================================

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    filtered_df = df.copy()

    with filter_col1:

        if "loan_grade" in df.columns:

            grades = ["All"] + sorted(
                df["loan_grade"].dropna().unique().tolist()
            )

            selected_grade = st.selectbox(
                "Loan Grade",
                grades
            )

            if selected_grade != "All":
                filtered_df = filtered_df[
                    filtered_df["loan_grade"] == selected_grade
                ]

    with filter_col2:

        if "person_home_ownership" in df.columns:

            ownership = ["All"] + sorted(
                df["person_home_ownership"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_home = st.selectbox(
                "Home Ownership",
                ownership
            )

            if selected_home != "All":
                filtered_df = filtered_df[
                    filtered_df["person_home_ownership"]
                    == selected_home
                ]

    with filter_col3:

        if "loan_intent" in df.columns:

            intents = ["All"] + sorted(
                df["loan_intent"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_intent = st.selectbox(
                "Loan Intent",
                intents
            )

            if selected_intent != "All":
                filtered_df = filtered_df[
                    filtered_df["loan_intent"]
                    == selected_intent
                ]

    st.divider()

    # ========================================================
    # FILTERED KPIs
    # ========================================================

    filtered_borrowers = len(filtered_df)

    filtered_default_rate = get_default_rate(filtered_df)

    filtered_avg_loan = (
        filtered_df["loan_amnt"].mean()
        if "loan_amnt" in filtered_df.columns
        and len(filtered_df) > 0
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Filtered Borrowers",
            f"{filtered_borrowers:,}"
        )

    with col2:
        st.metric(
            "Filtered Default Rate",
            f"{filtered_default_rate:.2f}%"
        )

    with col3:
        st.metric(
            "Average Loan",
            money(filtered_avg_loan)
        )

    st.divider()

    # ========================================================
    # FILTERED DATA
    # ========================================================

    st.subheader("📋 Filtered Borrowers")

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CREDIT RISK PREDICTOR
# ============================================================

elif page == "Credit Risk Predictor":

    st.header(" Individual Borrower Credit Risk Predictor")

    st.write(
        "Enter borrower information to estimate the probability of loan default."
    )

    st.divider()

    # ========================================================
    # BORROWER INPUTS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        person_age = st.number_input(
            "Borrower Age",
            min_value=18,
            max_value=100,
            value=30
        )

        person_income = st.number_input(
            "Annual Income ($)",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000
        )

        person_home_ownership = st.selectbox(
            "Home Ownership",
            sorted(
                df["person_home_ownership"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        person_emp_length = st.number_input(
            "Employment Length (Years)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5
        )

        loan_intent = st.selectbox(
            "Loan Intent",
            sorted(
                df["loan_intent"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            sorted(
                df["loan_grade"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        cb_person_default_on_file = st.selectbox(
            "Previous Default on File",
            sorted(
                df["cb_person_default_on_file"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with col2:

        loan_amnt = st.number_input(
            "Loan Amount ($)",
            min_value=0,
            max_value=1000000,
            value=10000,
            step=500
        )

        loan_int_rate = st.number_input(
            "Loan Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=11.0,
            step=0.1
        )

        loan_percent_income = st.number_input(
            "Loan Percent of Income",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01
        )

        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (Years)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5
        )

    st.divider()

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        " Predict Credit Risk",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        if loan_amnt > 0:

            income_per_loan = (
                person_income / loan_amnt
            )

        else:

            income_per_loan = 0

        if person_age > 0:

            credit_history_to_age = (
                cb_person_cred_hist_length / person_age
            )

        else:

            credit_history_to_age = 0

        # ----------------------------------------------------
        # CREATE BORROWER DATAFRAME
        # ----------------------------------------------------

        borrower = pd.DataFrame([{

            "person_age": person_age,

            "person_income": person_income,

            "person_home_ownership":
                person_home_ownership,

            "person_emp_length":
                person_emp_length,

            "loan_intent":
                loan_intent,

            "loan_grade":
                loan_grade,

            "loan_amnt":
                loan_amnt,

            "loan_int_rate":
                loan_int_rate,

            "loan_percent_income":
                loan_percent_income,

            "cb_person_default_on_file":
                cb_person_default_on_file,

            "cb_person_cred_hist_length":
                cb_person_cred_hist_length,

            "income_per_loan":
                income_per_loan,

            "credit_history_to_age":
                credit_history_to_age
        }])

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        try:

            prediction = model.predict(borrower)[0]

            probability = model.predict_proba(
                borrower
            )[0][1]

        except Exception as e:

            st.error(
                " Prediction failed."
            )

            st.exception(e)

            st.stop()

        # ----------------------------------------------------
        # RISK CLASSIFICATION
        # ----------------------------------------------------

        if probability < 0.30:

            risk = "Low Risk"

        elif probability < 0.60:

            risk = "Medium Risk"

        else:

            risk = "High Risk"

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader("📊 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Default Probability",
                f"{probability:.2%}"
            )

        with result_col2:

            st.metric(
                "Predicted Class",
                "Default" if prediction == 1
                else "Non-Default"
            )

        with result_col3:

            st.metric(
                "Risk Classification",
                risk
            )

        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        if risk == "Low Risk":

            st.success(
                "🟢 Low Risk — borrower has a relatively low estimated probability of default."
            )

        elif risk == "Medium Risk":

            st.warning(
                "🟡 Medium Risk — borrower requires additional credit assessment."
            )

        else:

            st.error(
                "🔴 High Risk — borrower has a relatively high estimated probability of default."
            )

        # ----------------------------------------------------
        # BORROWER DETAILS
        # ----------------------------------------------------

        with st.expander(" View Borrower Features"):

            st.dataframe(
                borrower.T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )


# ============================================================
# DATASET PREVIEW
# ============================================================

elif page == "Dataset Preview":

    st.header("🗃️ Credit Risk Dataset")

    st.write(
        "Preview of the processed dataset used in the project."
    )

    # Dataset information

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Columns",
            f"{df.shape[1]:,}"
        )

    with col3:
        st.metric(
            "Missing Values",
            f"{df.isna().sum().sum():,}"
        )

    st.divider()

    # Search

    search = st.text_input(
        "Search columns"
    )

    if search:

        matching_columns = [
            col for col in df.columns
            if search.lower() in col.lower()
        ]

        if matching_columns:

            st.write(
                "Matching columns:",
                matching_columns
            )

            st.dataframe(
                df[matching_columns].head(100),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No matching columns found."
            )

    else:

        st.dataframe(
            df.head(100),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # DATA TYPES
    # ========================================================

    st.subheader(" Dataset Structure")

    structure = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum().values
    })

    st.dataframe(
        structure,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Consumer Banking Credit Risk Pipeline | "
    "Data Analytics & Machine Learning Portfolio Project"
)

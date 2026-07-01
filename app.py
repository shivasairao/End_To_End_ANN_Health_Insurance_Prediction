"""
Streamlit app for the Medical Insurance High-Risk Prediction ANN.

Expects the following artifacts (produced by END_END_ANN_MEDICAL_INSURANCE.ipynb)
to sit in the same folder as this script:
    - medical_insurance_ann.keras
    - preprocessor.pkl
    - training_columns.pkl
    - best_hyperparameters.pkl   (optional, only shown in the sidebar)

Run with:
    streamlit run app.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Medical Insurance Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

ARTIFACT_DIR = Path(__file__).parent

MODEL_PATH = ARTIFACT_DIR / "medical_insurance_ann.keras"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
COLUMNS_PATH = ARTIFACT_DIR / "training_columns.pkl"
PARAMS_PATH = ARTIFACT_DIR / "best_hyperparameters.pkl"

CHRONIC_CONDITIONS = [
    "hypertension",
    "diabetes",
    "asthma",
    "copd",
    "cardiovascular_disease",
    "cancer_history",
    "kidney_disease",
    "liver_disease",
    "arthritis",
    "mental_health",
]


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    import tensorflow as tf

    missing = [p.name for p in [MODEL_PATH, PREPROCESSOR_PATH, COLUMNS_PATH] if not p.exists()]
    if missing:
        return None, None, None, None

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    with open(COLUMNS_PATH, "rb") as f:
        training_columns = pickle.load(f)

    best_params = None
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "rb") as f:
            best_params = pickle.load(f)

    return model, preprocessor, training_columns, best_params


def to_model_input(df: pd.DataFrame, preprocessor, training_columns):
    """Align columns to training order/names, then transform."""
    df = df.reindex(columns=training_columns)
    transformed = preprocessor.transform(df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return transformed


def predict(model, preprocessor, training_columns, df: pd.DataFrame, threshold: float):
    X = to_model_input(df, preprocessor, training_columns)
    proba = model.predict(X, verbose=0).ravel()
    label = (proba > threshold).astype(int)
    return proba, label


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🩺 Risk Predictor")
mode = st.sidebar.radio("Mode", ["Single patient", "Batch CSV upload"])
threshold = st.sidebar.slider(
    "Decision threshold (probability -> High Risk)", 0.05, 0.95, 0.50, 0.05
)

model, preprocessor, training_columns, best_params = load_artifacts()

if model is None:
    st.sidebar.error(
        "Model artifacts not found. Run the notebook first so that "
        "`medical_insurance_ann.keras`, `preprocessor.pkl`, and "
        "`training_columns.pkl` are saved next to this app."
    )

if best_params:
    with st.sidebar.expander("Best hyperparameters (from Optuna)"):
        st.json(best_params)

st.title("Medical Insurance — High-Risk Patient Predictor")
st.caption(
    "Artificial Neural Network trained on the medical insurance dataset to flag "
    "policyholders likely to be classified as high-risk."
)

if model is None:
    st.stop()


# --------------------------------------------------------------------------
# Single patient mode
# --------------------------------------------------------------------------
if mode == "Single patient":
    with st.form("patient_form"):
        st.subheader("Demographics & Lifestyle")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 0, 120, 45)
        sex = c2.selectbox("Sex", ["Female", "Male", "Other"])
        region = c3.selectbox("Region", ["North", "South", "East", "West", "Central"])
        urban_rural = c4.selectbox("Urban/Rural", ["Urban", "Suburban", "Rural"])

        c1, c2, c3, c4 = st.columns(4)
        income = c1.number_input("Annual income", 0.0, 1_000_000.0, 45000.0, step=1000.0)
        education = c2.selectbox(
            "Education", ["No HS", "HS", "Some College", "Bachelors", "Masters", "Doctorate"]
        )
        marital_status = c3.selectbox("Marital status", ["Single", "Married", "Divorced", "Widowed"])
        employment_status = c4.selectbox(
            "Employment status", ["Employed", "Self-employed", "Unemployed", "Retired"]
        )

        c1, c2, c3, c4 = st.columns(4)
        household_size = c1.number_input("Household size", 1, 15, 3)
        dependents = c2.number_input("Dependents", 0, 15, 1)
        smoker = c3.selectbox("Smoker", ["Never", "Former", "Current"])
        alcohol_freq = c4.selectbox("Alcohol frequency", ["None", "Occasional", "Weekly", "Daily"])

        st.subheader("Vitals & Labs")
        c1, c2, c3, c4 = st.columns(4)
        bmi = c1.number_input("BMI", 10.0, 70.0, 27.0, step=0.1)
        systolic_bp = c2.number_input("Systolic BP", 70.0, 250.0, 122.0)
        diastolic_bp = c3.number_input("Diastolic BP", 40.0, 150.0, 78.0)
        ldl = c4.number_input("LDL cholesterol", 20.0, 300.0, 110.0)

        c1, c2 = st.columns(2)
        hba1c = c1.number_input("HbA1c", 3.0, 15.0, 5.4, step=0.1)
        medication_count = c2.number_input("Medication count", 0, 30, 1)

        st.subheader("Chronic Conditions")
        cond_cols = st.columns(5)
        condition_flags = {}
        for i, cond in enumerate(CHRONIC_CONDITIONS):
            label = cond.replace("_", " ").title()
            condition_flags[cond] = int(cond_cols[i % 5].checkbox(label))

        st.subheader("Insurance Plan")
        c1, c2, c3, c4 = st.columns(4)
        plan_type = c1.selectbox("Plan type", ["HMO", "PPO", "EPO", "POS"])
        network_tier = c2.selectbox("Network tier", ["Bronze", "Silver", "Gold", "Platinum"])
        deductible = c3.number_input("Deductible", 0, 20000, 1000, step=100)
        copay = c4.number_input("Copay", 0, 500, 20, step=5)

        c1, c2 = st.columns(2)
        policy_term_years = c1.number_input("Policy term (years)", 1, 40, 5)
        policy_changes_last_2yrs = c2.number_input("Policy changes (last 2 yrs)", 0, 10, 0)
        provider_quality = st.slider("Provider quality score", 0.0, 5.0, 3.5, 0.1)

        st.subheader("Utilization & Claims")
        c1, c2, c3 = st.columns(3)
        visits_last_year = c1.number_input("Visits last year", 0, 100, 2)
        hospitalizations_last_3yrs = c2.number_input("Hospitalizations (last 3 yrs)", 0, 50, 0)
        days_hospitalized_last_3yrs = c3.number_input("Days hospitalized (last 3 yrs)", 0, 365, 0)

        c1, c2, c3 = st.columns(3)
        claims_count = c1.number_input("Claims count", 0, 200, 1)
        avg_claim_amount = c2.number_input("Average claim amount", 0.0, 500000.0, 500.0, step=50.0)
        total_claims_paid = c3.number_input("Total claims paid", 0.0, 1_000_000.0, 500.0, step=50.0)

        c1, c2, c3 = st.columns(3)
        annual_medical_cost = c1.number_input("Annual medical cost", 0.0, 500000.0, 5000.0, step=100.0)
        annual_premium = c2.number_input("Annual premium", 0.0, 100000.0, 900.0, step=50.0)
        monthly_premium = c3.number_input("Monthly premium", 0.0, 10000.0, 75.0, step=10.0)

        st.subheader("Procedures")
        c1, c2, c3, c4, c5 = st.columns(5)
        proc_imaging_count = c1.number_input("Imaging", 0, 50, 0)
        proc_surgery_count = c2.number_input("Surgery", 0, 20, 0)
        proc_physio_count = c3.number_input("Physio", 0, 50, 0)
        proc_consult_count = c4.number_input("Consults", 0, 50, 1)
        proc_lab_count = c5.number_input("Labs", 0, 50, 1)
        had_major_procedure = int(st.checkbox("Had a major procedure"))

        submitted = st.form_submit_button("Predict risk", use_container_width=True)

    if submitted:
        chronic_count = sum(condition_flags.values())

        row = {
            "age": age,
            "sex": sex,
            "region": region,
            "urban_rural": urban_rural,
            "income": income,
            "education": education,
            "marital_status": marital_status,
            "employment_status": employment_status,
            "household_size": household_size,
            "dependents": dependents,
            "bmi": bmi,
            "smoker": smoker,
            "alcohol_freq": alcohol_freq,
            "visits_last_year": visits_last_year,
            "hospitalizations_last_3yrs": hospitalizations_last_3yrs,
            "days_hospitalized_last_3yrs": days_hospitalized_last_3yrs,
            "medication_count": medication_count,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "ldl": ldl,
            "hba1c": hba1c,
            "plan_type": plan_type,
            "network_tier": network_tier,
            "deductible": deductible,
            "copay": copay,
            "policy_term_years": policy_term_years,
            "policy_changes_last_2yrs": policy_changes_last_2yrs,
            "provider_quality": provider_quality,
            "annual_medical_cost": annual_medical_cost,
            "annual_premium": annual_premium,
            "monthly_premium": monthly_premium,
            "claims_count": claims_count,
            "avg_claim_amount": avg_claim_amount,
            "total_claims_paid": total_claims_paid,
            "chronic_count": chronic_count,
            "proc_imaging_count": proc_imaging_count,
            "proc_surgery_count": proc_surgery_count,
            "proc_physio_count": proc_physio_count,
            "proc_consult_count": proc_consult_count,
            "proc_lab_count": proc_lab_count,
            "had_major_procedure": had_major_procedure,
            **condition_flags,
        }

        df_row = pd.DataFrame([row])
        proba, label = predict(model, preprocessor, training_columns, df_row, threshold)

        st.divider()
        result_col, gauge_col = st.columns([1, 2])
        with result_col:
            if label[0] == 1:
                st.error(f"### High Risk\nProbability: **{proba[0]:.1%}**")
            else:
                st.success(f"### Low Risk\nProbability: **{proba[0]:.1%}**")
            st.metric("Chronic conditions flagged", chronic_count)

        with gauge_col:
            st.write("Predicted probability of high risk")
            st.progress(min(max(float(proba[0]), 0.0), 1.0))
            st.caption(f"Decision threshold currently set to {threshold:.0%}")


# --------------------------------------------------------------------------
# Batch CSV mode
# --------------------------------------------------------------------------
else:
    st.subheader("Batch prediction from CSV")
    st.caption(
        "Upload a CSV with the same columns used in training "
        "(same schema as `medical_insurance.csv`, minus `is_high_risk`, `risk_score`, and `person_id`)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)

        missing_cols = [c for c in training_columns if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {missing_cols}")
        else:
            if "alcohol_freq" in batch_df.columns:
                batch_df["alcohol_freq"] = batch_df["alcohol_freq"].fillna("None")

            proba, label = predict(model, preprocessor, training_columns, batch_df, threshold)

            results = batch_df.copy()
            results["predicted_probability"] = proba
            results["predicted_high_risk"] = label

            st.success(f"Scored {len(results)} rows.")
            st.dataframe(results.head(50), use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Predicted high-risk", int(label.sum()))
            c2.metric("Predicted low-risk", int((1 - label).sum()))

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scored CSV",
                data=csv_bytes,
                file_name="medical_insurance_predictions.csv",
                mime="text/csv",
            )

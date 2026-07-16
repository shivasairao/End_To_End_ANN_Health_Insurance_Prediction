End-to-End ANN Health Insurance Risk Prediction

An end-to-end deep learning project that predicts whether a health insurance
policyholder is high-risk using an Artificial Neural Network (ANN), with
class-imbalance handling, automated hyperparameter tuning, and a deployed
Streamlit web app for single and batch predictions.

Overview

TaskBinary classification — is_high_risk (0 = Low Risk, 1 = High Risk)Dataset100,000 rows × 54 columns, synthetic medical insurance dataClass balance~63.2% Low Risk / ~36.8% High RiskModelArtificial Neural Network (Keras/TensorFlow)Imbalance handlingSMOTE (train set only)Hyperparameter tuningOptuna (15 trials, TPE sampler)DeploymentStreamlit Community Cloud

Dataset

The dataset (medical_insurance.csv) contains demographic, lifestyle, vitals,
insurance plan, and claims/utilization data per policyholder, including:


Demographics & lifestyle — age, sex, region, income, education, marital
status, employment, smoker/alcohol status
Vitals & labs — BMI, systolic/diastolic BP, LDL cholesterol, HbA1c
Chronic conditions — hypertension, diabetes, asthma, COPD, cardiovascular
disease, cancer history, kidney/liver disease, arthritis, mental health
Insurance plan — plan type, network tier, deductible, copay, policy term
Utilization & claims — visits, hospitalizations, claims count/amounts,
procedure counts


Two columns are dropped before training:


risk_score — correlates ~0.82 with the target and is almost certainly the
score the label was derived from, so it's excluded to avoid data leakage.
person_id — an identifier with no predictive value.


Pipeline


EDA & cleaning — missing values in alcohol_freq (~30k rows) filled
with "None"; duplicate rows dropped.
Leakage check — risk_score and person_id dropped (see above).
Train / validation / test split — 64% / 16% / 20%, stratified on the
target, random_state=21.
Preprocessing — numeric columns scaled with StandardScaler,
categorical columns one-hot encoded with OneHotEncoder, combined via a
ColumnTransformer fit only on the training set.
Baseline ANN — a simple 2-hidden-layer network (64 → 32 → 1, ReLU/
sigmoid) trained with early stopping, to sanity-check the pipeline.
Class imbalance (SMOTE) — applied to the training set only, after
preprocessing, to oversample the High Risk class.
Hyperparameter tuning (Optuna) — 15 trials searching over learning
rate, number of layers, units/layer, dropout, L1/L2 regularization,
optimizer (Adam/RMSprop/SGD), activation, and batch size, minimizing
validation loss.
Final model — rebuilt programmatically from study.best_params
(so re-running tuning automatically carries through), trained on the
SMOTE-resampled training set, validated on the untouched validation set.
Evaluation — accuracy, recall, classification report, and confusion
matrix on the held-out test set.
Artifacts saved — trained model, fitted preprocessor, training column
order, and best hyperparameters, so new records can be transformed and
scored identically at inference time.


Repository structure

.
├── app.py                              # Streamlit app (single + batch prediction)
├── END_END_ANN_MEDICAL_INSURANCE.ipynb # Full training notebook
├── end_end_ann_medical_insurance.py    # Notebook exported as a script
├── medical_insurance.csv               # Training dataset
├── medical_insurance_ann.keras         # Trained model
├── preprocessor.pkl                    # Fitted ColumnTransformer
├── training_columns.pkl                # Column order expected at inference
├── best_hyperparameters.pkl            # Optuna's best trial parameters
├── requirements.txt                    # Pinned, deployment-scoped dependencies
├── runtime.txt                         # Pins Python 3.11 for Streamlit Cloud
└── README.md

Running locally

bashgit clone https://github.com/shivasairao/End_To_End_ANN_Health_Insurance_Prediction.git
cd End_To_End_ANN_Health_Insurance_Prediction
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
streamlit run app.py

The app expects medical_insurance_ann.keras, preprocessor.pkl, and
training_columns.pkl to sit alongside app.py (already included in this
repo). best_hyperparameters.pkl is optional and only shown in the sidebar
for reference.

App features


Single patient mode — a form covering demographics, vitals, chronic
conditions, insurance plan, and utilization/claims history; returns a
High/Low Risk label with predicted probability.
Batch CSV mode — upload a CSV matching the training schema (minus
is_high_risk, risk_score, and person_id) to score many records at
once, view results inline, and download the scored file.
Adjustable decision threshold — a sidebar slider to move the
probability cutoff used to classify High Risk vs. Low Risk.


Deployment notes

Deployed on Streamlit Community Cloud. Two files matter for a clean
build:


runtime.txt — pins Python to 3.11. Without it, Streamlit Cloud
defaults to the newest available Python, which doesn't yet have stable
wheels for TensorFlow and related packages.
requirements.txt — kept minimal and scoped to what app.py actually
imports (streamlit, pandas, numpy, scikit-learn, tensorflow-cpu,
protobuf), rather than a full local-environment dependency dump, to avoid
version conflicts and platform-specific packages that don't exist on Linux.


Tech stack

Python · TensorFlow / Keras · scikit-learn · imbalanced-learn (SMOTE) ·
Optuna · Pandas · NumPy · Streamlit

Author

Aparaju Shivasai Rao
GitHub ·
LinkedIn

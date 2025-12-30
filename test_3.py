import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sqlite3
import os

# === Database helpers ===
def get_connection():
    return sqlite3.connect('patient_predictions.db', check_same_thread=False)

def save_patient_to_db(patient_dict, risk_label):
    conn = get_connection()
    cols = ', '.join([f'"{c}" TEXT' for c in patient_dict.keys()])
    create_sql = f'CREATE TABLE IF NOT EXISTS patients ({cols}, "risk_label" TEXT)'
    conn.execute(create_sql)
    columns = ', '.join([f'"{c}"' for c in patient_dict.keys()])
    placeholders = ', '.join(['?'] * (len(patient_dict) + 1))
    values = list(patient_dict.values()) + [risk_label]
    insert_sql = f'INSERT INTO patients ({columns}, "risk_label") VALUES ({placeholders})'
    conn.execute(insert_sql, values)
    conn.commit()
    conn.close()

@st.cache_resource
def load_artifacts():
    selected_features = joblib.load('C:/Users/harik/Downloads/selected_features.pkl')  # must match your new features!
    cluster_rule_labels = joblib.load('C:/Users/harik/Downloads/cluster_rule_labels.pkl')
    kmeans = joblib.load('C:/Users/harik/Downloads/crc_kmeans.pkl')
    scaler_path = 'C:/Users/harik/Downloads/my_scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        scaler_ok = True
    else:
        scaler = None
        scaler_ok = False
    latent_dim = 8
    class Autoencoder(torch.nn.Module):
        def __init__(self, input_dim, latent_dim=8):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, latent_dim)
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, input_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out
    ae = Autoencoder(len(selected_features), latent_dim)
    ae.load_state_dict(torch.load('C:/Users/harik/Downloads/crc_autoencoder.pt', map_location='cpu'))
    ae.eval()
    return selected_features, cluster_rule_labels, kmeans, scaler, ae, scaler_ok

# --- Preprocessing function (matches your new batch script) ---
def manual_row_preprocess(input_row, selected_features, scaler=None):
    df = pd.DataFrame([input_row])

    # Label encoding for nominal columns (only if present)
    label_encode_cols = ["Sex", "Race Category", "Smoker Status", "Smoking history"]
    for col in label_encode_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Mutation columns to binary (0/1)
    genetic_cols = [
        "KRAS", "TP53", "SDC2", "APC", "MSH2", "MSH6", "MLH1", "TIMP1",
        "KRAS: MUT", "KRAS: AMP", "KRAS: HOMDEL", "KRAS: FUSION",
        "TP53: MUT", "TP53: AMP", "TP53: HOMDEL", "TP53: FUSION",
        "SDC2: MUT", "SDC2: AMP", "SDC2: HOMDEL", "SDC2: FUSION",
        "APC: MUT", "APC: AMP", "APC: HOMDEL", "APC: FUSION",
        "MLH1: MUT", "MLH1: AMP", "MLH1: HOMDEL", "MLH1: FUSION",
        "MSH2: MUT", "MSH2: AMP", "MSH2: HOMDEL", "MSH2: FUSION",
        "MSH6: MUT", "MSH6: AMP", "MSH6: HOMDEL", "MSH6: FUSION",
        "TIMP1: MUT", "TIMP1: AMP", "TIMP1: HOMDEL", "TIMP1: FUSION"
    ]
    for col in genetic_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0 if str(x).strip().lower() in
                                    ["no alteration", "not profiled", "0", "none"] else 1)

    # Scale only "Age at Diagnosis" and "BMI"
    continuous_cols = [
        "Age at Diagnosis", "BMI"
    ]
    if scaler is not None:
        for col in continuous_cols:
            if col not in df.columns:
                df[col] = 0
        df[continuous_cols] = scaler.transform(df[continuous_cols])
    else:
        st.warning("Scaler not present! Results may be inaccurate.")

    df.fillna(0, inplace=True)
    df_model = df[selected_features]
    return df_model

# --- UI config ---
selected_features, cluster_rule_labels, kmeans, scaler, ae, scaler_ok = load_artifacts()
st.title("Individual Patient CRC Risk Prediction")

if not scaler_ok:
    st.warning("Scaler file not found! For best results, fit StandardScaler on your full training data and save to 'my_scaler.pkl'.")

label_encode_cols = ["Sex", "Race Category", "Smoker Status", "Smoking history"]
label_encoder_choices = {
    "Sex": ['Male', 'Female', 'Other'],
    "Race Category": ['White', 'Black', 'Asian', 'Other'],
    "Smoker Status": ['Never', 'Former', 'Current', 'Unknown'],
    "Smoking history": ['Yes', 'No', 'Unknown'],
}
genetic_cols = [
    "KRAS", "TP53", "SDC2", "APC", "MSH2", "MSH6", "MLH1", "TIMP1",
    "KRAS: MUT", "KRAS: AMP", "KRAS: HOMDEL", "KRAS: FUSION",
    "TP53: MUT", "TP53: AMP", "TP53: HOMDEL", "TP53: FUSION",
    "SDC2: MUT", "SDC2: AMP", "SDC2: HOMDEL", "SDC2: FUSION",
    "APC: MUT", "APC: AMP", "APC: HOMDEL", "APC: FUSION",
    "MLH1: MUT", "MLH1: AMP", "MLH1: HOMDEL", "MLH1: FUSION",
    "MSH2: MUT", "MSH2: AMP", "MSH2: HOMDEL", "MSH2: FUSION",
    "MSH6: MUT", "MSH6: AMP", "MSH6: HOMDEL", "MSH6: FUSION",
    "TIMP1: MUT", "TIMP1: AMP", "TIMP1: HOMDEL", "TIMP1: FUSION"
]
continuous_cols = [
    "Age at Diagnosis", "BMI"
]

if "user_inputs" not in st.session_state:
    st.session_state["user_inputs"] = {}

with st.form("patient_form", clear_on_submit=False):
    st.header("Fill in patient details")
    row = {}
    for feat in selected_features:
        if feat in label_encode_cols:
            val_list = label_encoder_choices.get(feat, ['Unknown'])
            row[feat] = st.selectbox(feat, val_list, key=feat)
        elif feat in genetic_cols:
            row[feat] = st.selectbox(feat, ["No alteration", "Mutation/Other"], key=feat)
        elif feat in continuous_cols:
            row[feat] = st.number_input(feat, value=0.0, key=feat)
        else:
            row[feat] = st.text_input(feat, "", key=feat)
    submit = st.form_submit_button("Predict Risk Group")
    reset = st.form_submit_button("Reset Form")

if reset:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

if submit:
    df_preprocessed = manual_row_preprocess(row, selected_features, scaler)
    with torch.no_grad():
        Z = ae.encoder(torch.tensor(df_preprocessed.values.astype(np.float32))).cpu().numpy()
    cluster = kmeans.predict(Z)
    risk_label = cluster_rule_labels[cluster[0]]
    st.success(f"Predicted Custom Risk Group: {risk_label}")

    # Download as CSV
    result_df = pd.DataFrame([row])
    result_df['Predicted Custom Risk Group'] = [risk_label]
    csv = result_df.to_csv(index=False).encode()
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="patient_prediction.csv",
        mime="text/csv"
    )
    save_patient_to_db(row, risk_label)
    st.info("Prediction saved to local database (patient_predictions.db in app directory).")

st.info("Gene mutation columns: select 'Mutation/Other' if alteration is present; else 'No alteration'.")

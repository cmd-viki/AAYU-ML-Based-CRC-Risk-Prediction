import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from pymongo import MongoClient
import datetime
import os

# ===== Page Config =====
st.set_page_config(page_title="AAYU CRC Risk Predictor", layout="wide")

# ===== Background & Style =====
st.markdown("""
<style>
.stApp {
    background-image: url("https://media.istockphoto.com/id/1297146235/photo/blue-chromosome-dna-and-gradually-glowing-flicker-light-matter-chemical-when-camera-moving.jpg?s=612x612&w=0&k=20&c=yjSdodXRBvtwzOYtQTqetnn3b4wWDNpF6keupxqxric=");
    background-size: cover;
    background-attachment: fixed;
    color: #ffffff;
}
h1, h2, h3, h4, h5, h6, p, label, .stRadio, .stSlider, .stTextInput, .stDownloadButton, .stButton, .stMarkdown, .stSelectbox {
    color: #ffffff !important;
}
.stButton > button {
    background: #1741b1;
    color: #fff;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 1.20em;
    margin-bottom: 10px;
    width: 100%;
    transition: background 0.3s;
}
.stButton > button:hover {
    background: #2161db;
}
.result-box {
    background: rgba(234, 246, 255, 0.85);
    color: #0b2545;
    border-radius: 14px;
    padding: 20px;
    margin-top: 20px;
}
.sidebar-title {
    font-size: 24px; 
    font-weight: bold; 
    color: #4da6ff;
}
</style>
""", unsafe_allow_html=True)

# ===== MongoDB Setup =====
@st.cache_resource
def get_mongo_collection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crc_patient_db"]
    return db["patients"]

collection = get_mongo_collection()

def save_to_mongo(data):
    data["timestamp"] = datetime.datetime.now()
    collection.insert_one(data)

# ===== Load Model & Artifacts =====
@st.cache_resource
def load_artifacts():
    selected_features = joblib.load('C:/Users/harik/Downloads/selected_features.pkl')
    kmeans = joblib.load('C:/Users/harik/Downloads/crc_kmeans.pkl')
    scaler = joblib.load('C:/Users/harik/Downloads/my_scaler.pkl')
    cluster_labels = joblib.load('C:/Users/harik/Downloads/cluster_rule_labels.pkl')
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
    ae.load_state_dict(torch.load('C:/Users/harik/Downloads/crc_autoencoder.pt', map_location=torch.device('cpu')))
    ae.eval()

    return selected_features, kmeans, scaler, cluster_labels, ae

selected_features, kmeans, scaler, cluster_labels, ae = load_artifacts()

# ===== Navigation =====
if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"

with st.sidebar:
    st.markdown("<div class='sidebar-title'>AAYU CRC Predictor</div>", unsafe_allow_html=True)
    if st.button("🏠 Home"):
        st.session_state["nav"] = "Home"
    if st.button("🧬 Risk Calculator"):
        st.session_state["nav"] = "Calculator"
    if st.button("📬 Contact / Feedback"):
        st.session_state["nav"] = "Contact"

# ===== Pages =====
nav = st.session_state["nav"]

# ===== Home Page =====
if nav == "Home":
    st.markdown("""
    <div style='margin-top:70px; padding:25px; background:rgba(0,0,0,0.6); color:#ffffff; border-radius:12px; max-width:700px; margin:auto;'>
        <h2>About AAYU</h2>
        <p><b>AAYU</b> is a machine learning–powered solution for comprehensive colorectal cancer (CRC) risk assessment.</p>
        <ul>
            <li>Interactive risk calculator for patients & clinicians</li>
            <li>Real-time dynamic predictions using Deep Learning</li>
            <li>Evidence-based, transparent risk interpretations</li>
        </ul>
        <p style='color:#ffd23f;'><b>Benefits:</b></p>
        <ul>
            <li>Facilitates early identification and targeted prevention of colorectal cancer.</li>
            <li>Advances personalized medicine and improves CRC outcomes.</li>
        </ul>      
        <p style='color:#ffd23f;'><b>Empower early action in colorectal health.</b></p>
    </div>
    """, unsafe_allow_html=True)

# ===== Risk Calculator =====
elif nav == "Calculator":
    st.header("🧬 CRC Risk Calculator")

    with st.form("risk_form"):
        patient_id = st.text_input("Patient ID", placeholder="Enter unique patient ID")
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age at Diagnosis", 20, 90, 50)
            bmi = st.slider("BMI", 15, 40, 24)
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

        with col2:
            diabetes = st.radio("Diabetes Mellitus History", ["No", "Yes"], horizontal=True)
            hypertension = st.radio("Hypertension History", ["No", "Yes"], horizontal=True)

        st.markdown("#### Genetic Biomarkers")
        cols = st.columns(4)
        genes = ["KRAS", "TP53", "APC", "SDC2", "MLH1", "MSH2", "MSH6", "PMS2", "TIMP1"]
        gene_input = {}
        for idx, gene in enumerate(genes):
            gene_input[gene + ": MUT"] = cols[idx % 4].radio(f"{gene} Mutation", [0, 1], horizontal=True)

        submit = st.form_submit_button("🚀 Predict Risk")

    if submit:
        if not patient_id.strip():
            st.error("⚠️ Please enter a valid Patient ID.")
        else:
            input_data = {
                "Patient ID": patient_id,
                "Age at Diagnosis": age,
                "BMI": bmi,
                "Sex": 0 if sex == "Male" else 1,
                "Diabetes Mellitus History": 1 if diabetes == "Yes" else 0,
                "Hypertension History": 1 if hypertension == "Yes" else 0
            }
            input_data.update(gene_input)

            for feat in selected_features:
                if feat not in input_data:
                    input_data[feat] = 0

            df = pd.DataFrame([input_data])
            df[["Age at Diagnosis", "BMI"]] = scaler.transform(df[["Age at Diagnosis", "BMI"]])
            x_input = torch.tensor(df[selected_features].values.astype(np.float32))

            with torch.no_grad():
                latent = ae.encoder(x_input).numpy()

            cluster = kmeans.predict(latent)[0]
            descriptive_label = cluster_labels[cluster]

            # Risk Mapping
            risk_map = {
                "Hereditary/High Risk (MMR, Young)": "High Risk",
                "High Oncogenic Risk (Multi-gene)": "High Risk",
                "High SDC2-Mediated Risk": "Moderate Risk",
                "Elevated Genetic Risk": "Moderate Risk",
                "Sporadic, Older Onset": "Low Risk",
                "Predominantly Sporadic": "Low Risk"
            }
            risk_group = risk_map.get(descriptive_label, "Unknown")

            # Display
            st.markdown(f"""
            <div class="result-box">
            <h4>🩺 Predicted Risk:</h4>
            <b>Patient ID:</b> {patient_id}<br>
            <b>Descriptive Cluster:</b> {descriptive_label}<br>
            <b>Final Risk Group:</b> {risk_group}
            </div>
            """, unsafe_allow_html=True)

            # Save to MongoDB
            to_save = input_data.copy()
            to_save["Descriptive Cluster"] = descriptive_label
            to_save["Risk Group"] = risk_group
            save_to_mongo(to_save)

            # Append to Master CSV
            master_file = "all_patients_crc_results.csv"
            df_out = pd.DataFrame([to_save])
            if os.path.exists(master_file):
                df_out.to_csv(master_file, mode='a', header=False, index=False)
            else:
                df_out.to_csv(master_file, index=False)

            # Download latest result
            st.download_button(
                label="⬇️ Download Latest Result CSV",
                data=df_out.to_csv(index=False).encode(),
                file_name=f"crc_risk_result_{patient_id}.csv",
                mime="text/csv"
            )

# ===== Contact & Feedback =====
elif nav == "Contact":
    st.header("📬 Contact & Feedback")
    feedback = st.text_area("Your Feedback", "Write here...", height=150)

    if st.button("Submit Feedback"):
        if feedback.strip():
            save_to_mongo({"feedback": feedback.strip()})
            st.success("Thank you for your feedback! 🌟")
        else:
            st.warning("Please write some feedback before submitting.")

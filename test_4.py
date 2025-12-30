import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from pymongo import MongoClient

# === Page Config ===
st.set_page_config(page_title="AAYU CRC Risk Predictor", layout="wide")

# === Persistent Background CSS ===
st.markdown("""
<style>
.stApp {
    background-image: url("https://media.istockphoto.com/id/1297146235/photo/blue-chromosome-dna-and-gradually-glowing-flicker-light-matter-chemical-when-camera-moving.jpg?s=612x612&w=0&k=20&c=yjSdodXRBvtwzOYtQTqetnn3b4wWDNpF6keupxqxric=");
    background-size: cover;
    background-attachment: fixed;
    color: #ffffff;
}
h1, h2, h3, h4, h5, h6, p, label, .stRadio, .stSlider, .stTextInput, .stDownloadButton, .stButton, .stMarkdown, .stSelectbox, .stDataFrame {
    color: #ffffff !important;
}
.stButton > button {
    width: 90%;
    background: #1741b1;
    color: #fff;
    padding: 12px 14px;
    border-radius: 8px;
    font-size: 1.06em;
    margin-bottom: 10px;
    transition: background 0.3s;
    border: none;
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
</style>
""", unsafe_allow_html=True)

# === MongoDB Setup ===
@st.cache_resource
def get_mongo_connection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crc_patient_db"]
    collection = db["patients"]
    return collection

def save_to_mongo(data):
    collection = get_mongo_connection()
    collection.insert_one(data)

# === Load Models ===
@st.cache_resource
def load_artifacts():
    selected_features = joblib.load('C:/Users/harik/Downloads/selected_features.pkl')
    kmeans = joblib.load('C:/Users/harik/Downloads/crc_kmeans.pkl')
    scaler = joblib.load('C:/Users/harik/Downloads/my_scaler.pkl')
    cluster_rule_labels = joblib.load('C:/Users/harik/Downloads/cluster_rule_labels.pkl')
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

    return selected_features, kmeans, scaler, cluster_rule_labels, ae

selected_features, kmeans, scaler, cluster_rule_labels, ae = load_artifacts()

# === Navigation State ===
if 'nav' not in st.session_state:
    st.session_state['nav'] = "Home"

# === Sidebar Navigation ===
with st.sidebar:
    st.markdown("## ")
    if st.button("Home"):
        st.session_state['nav'] = "Home"
    if st.button("Risk Calculator"):
        st.session_state['nav'] = "Calculator"
    if st.button("Contact / Feedback"):
        st.session_state['nav'] = "Contact"

# === Pages ===
nav = st.session_state['nav']

if nav == "Home":
    st.markdown("""
    <div style='margin-top:70px; padding:25px; background:rgba(0,0,0,0.6); color:#ffffff; border-radius:12px; max-width:600px; margin-left:auto; margin-right:auto;'>
        <h2>About AAYU</h2>
        <p><b>AAYU is a next-generation, machine learning–powered solution for comprehensive colorectal cancer (CRC) risk assessment. The platform blends modern AI techniques with current clinical science to provide actionable and personalized CRC risk stratification.</p>
        <ul>
            <li>Interactive risk calculator for patients & clinicians</li>
            <li>Real-time dynamic predictions powered by deep learning</li>
            <li>Transparent, evidence-based risk interpretations</li>
            <li>No personal data leaves your system</li>
        </ul>
        <p style='color:#ffd23f;'><b>Benefits:</b></p>
        <ul>
            <li>Facilitates early identification and targeted prevention of colorectal cancer.</li>
            <li>Champions the advancement of personalized medicine and contributes to improved colorectal cancer outcomes.</li>
        </ul>      
        <p style='color:#ffd23f;'><b>Empower early action in colorectal health.</b></p>
    </div>
    """, unsafe_allow_html=True)

elif nav == "Calculator":
    st.header("🧬 CRC Risk Calculator")

    with st.form("calc_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age at Diagnosis", 20, 90, 50)
            bmi = st.slider("BMI", 15, 40, 24)
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

        with col2:
            diabetes = st.radio("Diabetes History", ["No", "Yes"], horizontal=True)
            hypertension = st.radio("Hypertension History", ["No", "Yes"], horizontal=True)

        st.markdown("#### Genetic Biomarkers")
        cols = st.columns(4)
        genes = ["KRAS", "TP53", "APC", "SDC2", "MLH1", "MSH2", "MSH6", "PMS2", "TIMP1"]
        gene_input = {}
        for idx, gene in enumerate(genes):
            gene_input[gene + ": MUT"] = cols[idx % 4].radio(f"{gene}: MUT", [0, 1], horizontal=True)

        submitted = st.form_submit_button("🚀 Predict Risk")

    if submitted:
        # Prepare input
        patient_data = {
            "Age at Diagnosis": age,
            "BMI": bmi,
            "Sex": 0 if sex == "Male" else 1,
            "Diabetes Mellitus History": 1 if diabetes == "Yes" else 0,
            "Hypertension History": 1 if hypertension == "Yes" else 0
        }
        patient_data.update(gene_input)

        for feat in selected_features:
            if feat not in patient_data:
                patient_data[feat] = 0

        df_input = pd.DataFrame([patient_data])
        df_input[["Age at Diagnosis", "BMI"]] = scaler.transform(df_input[["Age at Diagnosis", "BMI"]])
        x_input = torch.tensor(df_input[selected_features].values.astype(np.float32))

        with torch.no_grad():
            latent = ae.encoder(x_input).numpy()

        cluster = kmeans.predict(latent)[0]
        descriptive_label = cluster_rule_labels[cluster]

        simplified_map = {
            "Hereditary/High Risk (MMR, Young)": "High Risk",
            "High Oncogenic Risk (Multi-gene)": "High Risk",
            "High SDC2-Mediated Risk": "Moderate Risk",
            "Elevated Genetic Risk": "Moderate Risk",
            "Sporadic, Older Onset": "Low Risk",
            "Predominantly Sporadic": "Low Risk"
        }
        final_label = simplified_map.get(descriptive_label, "Unknown")

        # Display Result
        st.markdown(f"""
        <div class="result-box">
        <h4>🩺 Predicted Risk:</h4>
        <b>Descriptive Cluster:</b> {descriptive_label}<br>
        <b>Final Risk Group:</b> {final_label}
        </div>
        """, unsafe_allow_html=True)

        # Save to MongoDB
        save_data = patient_data.copy()
        save_data["Descriptive Cluster"] = descriptive_label
        save_data["Final Risk Group"] = final_label
        save_to_mongo(save_data)

        # Downloadable CSV
        df_out = pd.DataFrame([save_data])
        csv = df_out.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download Result CSV",
            data=csv,
            file_name="crc_risk_result.csv",
            mime="text/csv"
        )

elif nav == "Contact":
    st.markdown("""
    <div style='margin-top:60px; padding:25px; background:rgba(0,0,0,0.6); color:#ffffff; border-radius:12px; max-width:500px; margin-left:auto; margin-right:auto;'>
        <h3>Contact & Feedback</h3>
        <p>We welcome your feedback!</p>
    </div>
    """, unsafe_allow_html=True)

    feedback = st.text_area("Your Feedback", "Write here...", height=100)
    if st.button("Submit Feedback"):
        if feedback.strip():
            st.success("Thank you for your feedback! 🌟")
        else:
            st.warning("Please enter some feedback before submitting.")

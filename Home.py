import streamlit as st
from streamlit_lottie import st_lottie
import requests

# --- Page Configuration ---
st.set_page_config(page_title="CRC Risk Predictor", page_icon="🧬", layout="wide")

# --- Lottie Loader ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Load Animation ---
lottie_virus = load_lottieurl("https://lottie.host/459faed3-1b55-4c10-8852-c8031a6d55ba/1FBAwMQ6ib.json")

# --- Background CSS ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://media.istockphoto.com/id/1297146235/photo/blue-chromosome-dna-and-gradually-glowing-flicker-light-matter-chemical-when-camera-moving.jpg?s=612x612&w=0&k=20&c=yjSdodXRBvtwzOYtQTqetnn3b4wWDNpF6keupxqxric=");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }}
    .navbar {{
        background-color: rgba(230, 57, 70, 0.95);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }}
    .navbar a {{
        color: black;
        margin: 0 20px;
        text-decoration: none;
        font-size: 20px;
        font-weight: bold;
    }}
    .btn-container {{
        margin-top: 30px;
    }}
    .btn {{
        background-color: white;
        color: #e63946;
        padding: 12px 25px;
        margin-right: 15px;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        font-size: 16px;
        transition: 0.3s;
    }}
    .btn:hover {{
        background-color: #f1f1f1;
        transform: scale(1.05);
    }}
    .about-section {{
        background-color: rgba(255, 255, 255, 0.9);
        color: #0c1c3c;
        padding: 30px;
        margin-top: 50px;
        border-radius: 15px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Navbar ---
st.markdown("""
<div class="navbar">
    <a href="#">HOME</a>
    <a href="#about">ABOUT</a>
    <a href="#risk">TAKE ACTION</a>
    <a href="#contact">CONTACT</a>
</div>
""", unsafe_allow_html=True)

# --- Main Headline ---
st.markdown("<br>", unsafe_allow_html=True)
st.title("Care Early, Predict CRC Risk")
st.subheader("AI-Powered Colorectal Cancer Risk Stratification")

# --- Animation & Buttons ---
col1, col2 = st.columns([2, 1])

with col1:
    if lottie_virus:
        st_lottie(lottie_virus, height=350, key="dna_lottie")

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="btn-container">
        <button class="btn" onclick="window.location.href='#risk';">Read More</button>
        <button class="btn" onclick="window.location.href='#about';">About Us</button>
    </div>
    """, unsafe_allow_html=True)

# --- About Section ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div id="about" class="about-section">
    <h2>About CRC Risk Predictor</h2>
    <p>
    CRC Risk Predictor is a medical AI platform that stratifies patients into risk groups based on clinical and genetic biomarkers for colorectal cancer.
    </p>
    <p>
    Using unsupervised machine learning and biomarker analysis, we provide early risk insights for better healthcare action.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Contact Section ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div id="contact" style="text-align:center; margin-top:50px;">
    <h4>Contact Us: aayu@crc-risk.ai</h4>
</div>
""", unsafe_allow_html=True)

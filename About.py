import streamlit as st

st.set_page_config(page_title="About Us", page_icon="ℹ️", layout="centered")

st.title("ℹ️ About Us")

st.markdown("""
### AAYU CRC Risk Predictor

This project is developed as part of the **"AI for Healthcare" initiative**.

**Key Highlights:**

- Predicts CRC risk using an **unsupervised learning pipeline**
- Combines **Autoencoders + Clustering + Clinical Interpretation**
- No manual labels required for risk prediction

---

### Team:

- **Project Lead**: Sritha  
- **ML Engineer**: [Your Name]  
- **Domain Expert**: [Advisor Name]

---

### Contact:

📧 [your_email@example.com](mailto:your_email@example.com)

🔗 **GitHub**: [github.com/yourrepo](https://github.com/yourrepo)

---
""")

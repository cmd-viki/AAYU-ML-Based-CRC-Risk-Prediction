# AAYU – Colorectal Cancer (CRC) Risk Stratification Suite

AAYU is a **machine-learning**–based toolkit for colorectal cancer (CRC) risk stratification, combining autoencoders, clustering, and explainable models with interactive Streamlit apps for clinicians and researchers.

## Repository structure

- `test.py` / `test_4.py` / `test_5.py`  
  Streamlit apps for CRC risk prediction using a trained autoencoder + KMeans clustering model and rule‑based risk labels, with MongoDB logging and CSV export.
- `crc_risk.py`  
  Streamlit app for CRC cluster interpretation and relabeling on uploaded patient cohorts, including autoencoder training, KMeans clustering, RandomForest feature importance, cluster profiling, and SHAP plots.[1]
- `crc_risk_metrics.py`  
  Offline script to evaluate clustering quality (Silhouette, Calinski‑Harabasz, Davies‑Bouldin, Adjusted Rand Index) and compute RandomForest feature importance on the autoencoder latent space.
- `model1.py`, `model5.py`, `test_3.py`  
  Additional model/experiment scripts (autoencoder + clustering + classification) for CRC risk modeling and ablation/testing.
- `PreProcess_modify2.py`  
  Preprocessing utilities to clean, encode and scale clinical + genomic features prior to autoencoder/clustering.
- `Home.py`, `About.py`  
  Streamlit pages for app landing/about content integrated into the AAYU UI.
- `CRC_Train_Set.xlsx`, `CRC_Validation_Set.xlsx`, `CRC_Test_Set.xlsx`  
  Structured CRC patient cohorts used for model training, validation and testing.
- `CBioPortal_DATA_no_missing.xlsx`, `CBioPortal_DATA_ready_for_autoencoder_FINAL.xlsx`  
  CBioPortal‑derived CRC datasets (cleaned and no‑missing) used for autoencoder training and clustering experiments.
- `all_patients_crc_results.csv`  
  Aggregated predictions saved from the Streamlit risk calculator for all processed patients.
- `patient_custom_risk_labels.csv`  
  Custom, human‑interpretable labels mapped to discovered CRC clusters.
- `crc_risk_clustering_metrics.csv`, `crc_risk_feature_importance.csv`  
  Saved clustering quality metrics and feature importance results from `crc_risk_metrics.py`.

## Core workflow

1. **Data preparation**  
   - Start from raw CBioPortal/CRC cohort files (clinical + genomic).  
   - Clean and encode features using `PreProcess_modify2.py`, then export to the `CBioPortal_DATA_ready_for_autoencoder_FINAL.xlsx` / CRC train–val–test Excel files.

2. **Model training and clustering (offline)**  
   - Train an autoencoder on the prepared feature set and obtain latent representations.  
   - Run KMeans on the latent space, compute cluster quality metrics, and derive feature importance with `crc_risk_metrics.py` (outputs the `crc_risk_clustering_metrics.csv` and `crc_risk_feature_importance.csv`).

3. **Cluster interpretation and relabeling**  
   - Use `crc_risk.py` to upload a CRC cohort, re‑fit an autoencoder and KMeans (or re‑use saved models), view cluster profiles, inspect top features via RF, and relabel clusters into clinically meaningful risk groups (Low / Moderate / High, etc.).

4. **Interactive risk prediction (deployed app)**  
   - Run one of the Streamlit apps (`test.py` / `test_4.py` / `test_5.py`) to get per‑patient risk.
   - The app:
     - Collects clinical data (age at diagnosis, BMI, sex, diabetes and hypertension history) and binary genomic biomarkers (KRAS, TP53, APC, SDC2, MLH1, MSH2, MSH6, PMS2, TIMP1).[3][5]
     - Uses pre‑trained artifacts (`selectedfeatures.pkl`, `crc_autoencoder.pt`, `crc_kmeans.pkl`, `myscaler.pkl`, `clusterrulelabels.pkl`) to encode the patient into latent space, assign a cluster, and map it to a final risk group (e.g., “High Oncogenic Risk – Multi‑gene High Risk”).
     - Logs results to MongoDB (`crcpatientdb.patients`) and appends to `all_patients_crc_results.csv`, with an option to download an individual CSV.

## Installation

1. **Clone repo**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. **Create and activate environment**

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

3. **Install dependencies**

Create a `requirements.txt` including (adjust versions as needed):

```text
streamlit
pandas
numpy
scikit-learn
tensorflow
torch
joblib
shap
seaborn
matplotlib
pymongo
openpyxl
```

Then run:

```bash
pip install -r requirements.txt
```

4. **Configure MongoDB (optional)**  
   - Start a local MongoDB instance (default: `mongodb://localhost:27017`).  
   - The apps expect a database `crcpatientdb` and collection `patients`; these are created automatically on first insert.

5. **Place model artifacts**  
   - Put the following files in an accessible path (or update paths in the code):  
     - `selectedfeatures.pkl`  
     - `crc_kmeans.pkl` (or `crckmeans.pkl`)  
     - `myscaler.pkl`  
     - `clusterrulelabels.pkl`  
     - `crc_autoencoder.pt

   Update hard‑coded paths (currently pointing to `C:/Users/harik/Downloads/...`) if needed.

## Running the applications

### 1. CRC risk prediction app

From the repo root:

```bash
streamlit run test.py
# or
streamlit run test_4.py
# or
streamlit run test_5.py
```

- Open the URL printed by Streamlit (usually `http://localhost:8501`).  
- Use the sidebar to navigate between **Home**, **Risk Calculator**, and **Contact/Feedback**.

### 2. Cluster interpretation & relabeling app

```bash
streamlit run crc_risk.py
```

- Upload a CRC cohort `.xlsx` file with the required clinical and genomic columns.  
- Tune autoencoder latent dimension and training epochs from the sidebar, then run clustering, inspect top features, relabel clusters, and download the relabeled CSV.

### 3. Clustering metrics and feature importance (offline)

```bash
python crc_risk_metrics.py
```

- This script loads the prepared CBioPortal/CRC dataset, encodes features, passes them through the trained autoencoder, runs KMeans, and prints metric values.  
- It generates `crc_risk_clustering_metrics.csv` and `crc_risk_feature_importance.csv` in the working directory.

## Expected input schema (apps)

The patient‑level apps expect:

- **Clinical**:  
  - `Age at Diagnosis` (years, numeric)  
  - `BMI` (numeric)  
  - `Sex` (Male/Female, encoded internally)  
  - `Diabetes Mellitus History` or `Diabetes History` (Yes/No)  
  - `Hypertension History` or `Hypertension History` (Yes/No)

- **Genomic (binary)**:  
  - `KRAS_MUT`, `TP53_MUT`, `APC_MUT`, `SDC2_MUT`, `MLH1_MUT`, `MSH2_MUT`, `MSH6_MUT`, `PMS2_MUT`, `TIMP1_MUT` (0 = no alteration, 1 = mutation/alteration).

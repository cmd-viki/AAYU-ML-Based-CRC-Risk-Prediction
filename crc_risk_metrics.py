import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# === Load data ===
data = pd.read_excel("C:/Users/harik/Downloads/CBioPortal_DATA_no_missing.xlsx")

# === Full feature list used during AE training ===
model_features = [
    "Age at Diagnosis", "BMI", "Sex", "Diabetes Mellitus History", "Hypertension History",
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

df = data[model_features].copy()
df.fillna(0, inplace=True)

# === Load models ===
kmeans = joblib.load('C:/Users/harik/Downloads/crc_kmeans.pkl')
scaler = joblib.load('C:/Users/harik/Downloads/my_scaler.pkl')

# Autoencoder
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

ae = Autoencoder(len(model_features), latent_dim)
ae.load_state_dict(torch.load('C:/Users/harik/Downloads/crc_autoencoder.pt', map_location='cpu'))
ae.eval()

# === Encoding ===
cat_cols = ["Sex", "Diabetes History", "Hypertension History"]
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

def encode_mut(x):
    if isinstance(x, str):
        if x.strip().lower() in ["no alteration", "not profiled", "none", "0"]:
            return 0
        else:
            return 1
    elif isinstance(x, (int, float)):
        return 0 if x == 0 else 1
    else:
        return 1

for col in model_features:
    if any(gene in col for gene in ["KRAS", "TP53", "SDC2", "APC", "MSH2", "MSH6", "MLH1", "TIMP1"]):
        df[col] = df[col].apply(encode_mut)

# === Scaling ===
continuous_cols = ["Age at Diagnosis", "BMI"]
if scaler is not None:
    df[continuous_cols] = scaler.transform(df[continuous_cols])

# === Autoencoder Latent Space ===
with torch.no_grad():
    X_tensor = torch.tensor(df.values.astype(np.float32))
    Z = ae.encoder(X_tensor).cpu().numpy()

# === Clustering ===
clusters = kmeans.predict(Z)

# === Clustering Metrics ===
sil_score = silhouette_score(Z, clusters)
ch_score = calinski_harabasz_score(Z, clusters)
db_score = davies_bouldin_score(Z, clusters)

print("\n=== Clustering Quality Metrics ===")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")

# === Feature Importance via Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df, clusters)
importances = rf.feature_importances_

feat_imp = pd.Series(importances, index=df.columns).sort_values(ascending=False)
print("\n=== Top Features Driving Clusters ===")
print(feat_imp.head(10))

# === Stability Check: Adjusted Rand Index ===
kmeans2 = KMeans(n_clusters=3, random_state=99).fit(Z)
clusters2 = kmeans2.labels_

ari = adjusted_rand_score(clusters, clusters2)
print("\n=== Stability Check ===")
print(f"Adjusted Rand Index (between cluster runs): {ari:.4f}")

# === Save Outputs ===
metrics = {
    "Silhouette Score": sil_score,
    "Calinski-Harabasz Index": ch_score,
    "Davies-Bouldin Index": db_score,
    "Adjusted Rand Index": ari
}
pd.Series(metrics).to_csv("crc_risk_clustering_metrics.csv")
feat_imp.to_csv("crc_risk_feature_importance.csv")

print("\n✅ Metrics and feature importance saved as CSV.")

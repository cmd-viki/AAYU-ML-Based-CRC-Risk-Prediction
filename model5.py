import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# LOAD TRAINING DATA
df = pd.read_excel("C:\\Users\\harik\\Desktop\\Mini_Project\\CRC_Train_Set.xlsx")

# SELECT CLINICAL & GENETIC FEATURES
base_features = [
    "Age at Diagnosis", "BMI", "Sex", "Diabetes Mellitus History", "Hypertension History"
]
genes = ['KRAS', 'TP53', 'SDC2', 'APC', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'TIMP1']
genetic_cols = [col for col in df.columns if any(gene in col for gene in genes)]
selected_features = base_features + genetic_cols
df_selected = df[selected_features].copy()

# AUTOENCODER
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

X = df_selected.values.astype(np.float32)
feature_names = df_selected.columns.tolist()
latent_dim = 8
ae = Autoencoder(X.shape[1], latent_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae = ae.to(device)
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
criterion = nn.MSELoss()
loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=32, shuffle=True)
ae.train()
for epoch in range(30):
    running_loss = 0.
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = ae(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(batch)
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss {running_loss / len(df_selected):.5f}')

# CLUSTERING
ae.eval()
with torch.no_grad():
    Z = ae.encoder(torch.tensor(X).to(device)).cpu().numpy()
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(Z)
df_selected['cluster'] = clusters

# FEATURE IMPORTANCE (optional)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, clusters)
importances = rf.feature_importances_
sorted_feat = sorted(zip(feature_names, importances), key=lambda x: -x[1])
print("\nMost important features driving clusters:")
for name, score in sorted_feat[:10]:
    print(f'{name}: {score:.3f}')

# RULE-BASED CLUSTER RISK ASSIGNMENT
mmr_cols = ["MLH1: MUT", "MSH2: MUT", "MSH6: MUT", "PMS2: MUT"]
kras_cols = ["KRAS: MUT"]
tp53_cols = ["TP53: MUT"]
apc_cols = ["APC: MUT"]
sdc2_cols = ["SDC2: MUT"]
cluster_rule_labels = {}
for clust in sorted(df_selected['cluster'].unique()):
    mask = (df_selected['cluster'] == clust)
    n_pat = mask.sum()
    mmr_mut = (df_selected.loc[mask, mmr_cols].sum(axis=1) > 0).mean() if all(col in df_selected for col in mmr_cols) else 0
    kras_mut = (df_selected.loc[mask, kras_cols].sum(axis=1) > 0).mean() if all(col in df_selected for col in kras_cols) else 0
    tp53_mut = (df_selected.loc[mask, tp53_cols].sum(axis=1) > 0).mean() if all(col in df_selected for col in tp53_cols) else 0
    apc_mut = (df_selected.loc[mask, apc_cols].sum(axis=1) > 0).mean() if all(col in df_selected for col in apc_cols) else 0
    sdc2_mut = (df_selected.loc[mask, sdc2_cols].sum(axis=1) > 0).mean() if all(col in df_selected for col in sdc2_cols) else 0
    age_under_50 = (df_selected.loc[mask, "Age at Diagnosis"] < 50).mean() if "Age at Diagnosis" in df_selected else 0
    age_over_50 = (df_selected.loc[mask, "Age at Diagnosis"] > 50).mean() if "Age at Diagnosis" in df_selected else 0
    if mmr_mut > 0.3 and age_under_50 > 0.2:
        label = "Hereditary/High Risk (MMR, Young)"
    elif kras_mut > 0.4 and tp53_mut > 0.4 and apc_mut > 0.4:
        label = "High Oncogenic Risk (Multi-gene)"
    elif sdc2_mut > 0.3:
        label = "High SDC2-Mediated Risk"
    elif mmr_mut > 0.2 or tp53_mut > 0.3:
        label = "Elevated Genetic Risk"
    elif age_over_50 > 0.7:
        label = "Sporadic, Older Onset"
    else:
        label = "Predominantly Sporadic"
    cluster_rule_labels[clust] = label
df_selected["custom_risk_group"] = df_selected["cluster"].map(cluster_rule_labels)

# OUTPUT patient risk labels (for train set)
df_selected['patient_id'] = df_selected.index
result = df_selected[['patient_id', 'custom_risk_group']]
result.to_csv("patient_custom_risk_labels.csv", index=False)
print(result)

# --- SAVE EVERYTHING NEEDED FOR TESTING ---
torch.save(ae.state_dict(), 'C:\\Users\\harik\\Downloads\\crc_autoencoder.pt')
joblib.dump(kmeans, 'C:\\Users\\harik\\Downloads\\crc_kmeans.pkl')
joblib.dump(cluster_rule_labels, 'C:\\Users\\harik\\Downloads\\cluster_rule_labels.pkl')
joblib.dump(selected_features, 'C:\\Users\\harik\\Downloads\\selected_features.pkl')

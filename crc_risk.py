import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import shap
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AAYU – CRC Cluster Relabeling", layout="wide")

st.title("🧬 AAYU – CRC Cluster Interpretation & Relabeling Pipeline")

uploaded_file = st.file_uploader("Upload your CRC patient data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.sidebar.header("⚙️ Model Parameters")
    encoding_dim = st.sidebar.slider("Latent Space Dimension (Autoencoder)", 2, 20, 10)
    epochs = st.sidebar.slider("Autoencoder Training Epochs", 50, 500, 150)
    cluster_count = 3

    # Preprocessing
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    id_col = None
    if 'Patient_ID' in df.columns:
        id_col = df['Patient_ID']
        cat_cols.remove('Patient_ID')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ✅ Safe Encoding
    if cat_cols:
        df_encoded = pd.get_dummies(df[cat_cols])
        X = pd.concat([df[num_cols], df_encoded], axis=1)
    else:
        df_encoded = pd.DataFrame()
        X = df[num_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Autoencoder
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=16, verbose=0)

    encoder_model = Model(inputs=input_layer, outputs=encoded)
    X_latent = encoder_model.predict(X_scaled)

    # Clustering
    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    clusters = kmeans.fit_predict(X_latent)

    df_clusters = pd.DataFrame()
    if id_col is not None:
        df_clusters['Patient_ID'] = id_col
    df_clusters['Cluster'] = clusters

    st.subheader("🧩 KMeans Clustering Done")
    st.dataframe(df_clusters.head())

    # Train Classifier to Predict Clusters (for explainability)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, clusters)

    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)

    st.subheader("🔍 Top Features Driving the Clusters")
    st.table(top_features)

    # Use top features to profile clusters and relabel
    X_copy = pd.DataFrame(X, columns=X.columns)
    X_copy['Cluster'] = clusters

    cluster_summary = X_copy.groupby('Cluster')[top_features.index].mean()

    st.subheader("📊 Cluster Profiles Based on Top Features")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(cluster_summary, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Relabel Clusters based on top feature
    main_feature = top_features.index[0]  # Take top feature for relabeling

    # Rank clusters by average of main_feature
    cluster_order = cluster_summary[main_feature].sort_values().index.tolist()

    relabel_mapping = {}
    for idx, cluster in enumerate(cluster_order):
        if idx == 0:
            relabel_mapping[cluster] = 'Low Risk'
        elif idx == 1:
            relabel_mapping[cluster] = 'Medium Risk'
        else:
            relabel_mapping[cluster] = 'High Risk'

    df_clusters['Assigned_Label'] = df_clusters['Cluster'].map(relabel_mapping)

    st.subheader("🏥 Final Relabeled Patient Risk Groups")
    st.dataframe(df_clusters)

    # Download output
    csv = df_clusters.to_csv(index=False)
    st.download_button("📥 Download Relabeled Output", csv, "AAYU_CRC_Risk_Relabeling.csv", "text/csv")

    # SHAP Interpretability
    st.subheader("🧠 SHAP Feature Importance (Interpretability)")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(bbox_inches='tight')

    st.success("✅ Clusters Interpreted & Relabeled Successfully!")

else:
    st.info("Please upload a dataset to start.")

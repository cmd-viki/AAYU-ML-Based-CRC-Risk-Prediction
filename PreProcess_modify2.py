import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# 🔹 Step 1: Load the CRC patient dataset from Excel
df = pd.read_excel("C:\\Users\\harik\\Desktop\\Mini_Project\\CBioPortal_DATA_no_missing.xlsx")

# 🔹 Step 2: Manually map ordinal categorical variables to meaningful numeric values
# These have an inherent order — so we preserve that using mapping


# 🔹 Step 3: Label encode nominal categorical variables (no order, just convert strings to numbers)
# These include sex, race, smoking status, etc.
label_encode_cols = ["Sex", "Race Category", "Smoker Status", "Smoking history",]
for col in label_encode_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 🔹 Step 4: Convert mutation-based genetic alteration columns to binary values
# 0 → No alteration / not profiled | 1 → Mutation present
genetic_cols = [
    "KRAS","TP53","SDC2","APC","MSH2","MSH6","MLH1","TIMP1",
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
        df[col] = df[col].apply(lambda x: 0 if str(x).strip().lower() in ["no alteration", "not profiled", "0", "none"] else 1)

# 🔹 Step 5: Normalize continuous features using StandardScaler (mean=0, std=1)
# This step ensures that features like age, TMB, BMI, etc., are on the same scale
continuous_cols = [
    "Age at Diagnosis", "BMI", 
   
]

scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
joblib.dump(scaler, "C:/Users/harik/Downloads/my_scaler.pkl")
print("Scaler saved!")


# 🔹 Step 6: Fill any missing values as a safety net (even if no missing values expected)
df.fillna(0, inplace=True)

# 🔹 Step 7: Save the processed dataframe to an Excel file for later modeling (autoencoders, clustering, etc.)
df.to_excel("C:\\Users\\harik\\Downloads\\CBioPortal_DATA_ready_for_autoencoder_FINAL.xlsx", index=False)

print("✅ Preprocessing complete. ALL mutation columns retained.")

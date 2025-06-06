import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

df = pd.read_csv("cleaned_healthcare_dataset.csv")

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

label_cols = ['Gender', 'Blood Type', 'Insurance Provider', 'Admission Type']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

if 'Test Results' in df.columns:
    df['Test Results'] = df['Test Results'].astype(str).str.strip()
    df['Test Results'] = LabelEncoder().fit_transform(df['Test Results'])

df = pd.get_dummies(df, columns=['Medical Condition'], drop_first=True)

scaler = StandardScaler()
numeric_cols = ['Age', 'Billing Amount', 'Length of Stay']
missing_cols = [col for col in numeric_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns for scaling: {missing_cols}")
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

apply_pca = True

if apply_pca:
    from numpy import isnan, isinf
    pca_input = df.select_dtypes(include='number').drop(columns=['Room Number', 'Test Results'], errors='ignore')
    mask = pca_input.apply(lambda row: row.map(lambda x: not (isnan(x) or isinf(x))).all(), axis=1)
    pca_input = pca_input[mask]
    df = df.loc[pca_input.index]
    pca = PCA(n_components=0.95)
    pca_result = pca.fit_transform(pca_input)
    pca_df = pd.DataFrame(pca_result, index=df.index)
    pca_df['Test Results'] = df['Test Results'].values
    preprocessed_df = pca_df
else:
    preprocessed_df = df.copy()

preprocessed_df.to_csv("preprocessed_healthcare_data.csv", index=False)
print("Preprocessing complete. Data is saved to a new csv called 'preprocessed_healthcare_data.csv'.")

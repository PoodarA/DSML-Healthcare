import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("cleaned_healthcare_dataset.csv")

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

label_cols = ['Gender', 'Blood Type', 'Insurance Provider', 'Admission Type']
for col in label_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

if 'Test Results' in df.columns:
    df['Test Results'] = LabelEncoder().fit_transform(df['Test Results'].astype(str))

if 'Medical Condition' in df.columns:
    df = pd.get_dummies(df, columns=['Medical Condition'], drop_first=True)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

features_for_pca = df.select_dtypes(include='number').drop(columns=['Room Number', 'Test Results'], errors='ignore')
features_for_pca = features_for_pca.loc[:, features_for_pca.std() > 0]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_pca)
scaled_features = np.clip(scaled_features, -1e6, 1e6)
scaled_features = scaled_features.astype(np.float32)

print("PCA input shape:", scaled_features.shape)
print("Any NaN:", np.isnan(scaled_features).any())
print("Any inf:", np.isinf(scaled_features).any())
print("Min value:", np.min(scaled_features))
print("Max value:", np.max(scaled_features))

pca = PCA(n_components=0.95)
pca_result = pca.fit_transform(scaled_features)

print(f"PCA reduced from {scaled_features.shape[1]} to {pca_result.shape[1]} components.")

pca_df = pd.DataFrame(pca_result, index=features_for_pca.index)

if 'Test Results' in df.columns:
    pca_df['Test Results'] = df.loc[pca_df.index, 'Test Results']

pca_df.to_csv("preprocessed_healthcare_data.csv", index=False)
print("Preprocessing complete. PCA data saved as 'preprocessed_healthcare_data.csv'.")

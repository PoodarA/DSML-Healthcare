"""
Healthcare Dataset Cleaning and Exploration
Authors: Alfarwq Sharif, Erza, Ruth
Course: Data Science and Machine Learning
Description:
    This script performs data cleaning and exploratory analysis on the Kaggle Healthcare Dataset.
    It includes basic visualizations and transformations to prepare for preprocessing and modeling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

# ===============================
# Step 1: Standardize 'Name' column
# ===============================
# Convert all names to uppercase for consistency
df['Name'] = df['Name'].str.upper()

# ===============================
# Step 2: Convert date columns
# ===============================
# Change 'Date of Admission' and 'Discharge Date' to datetime format
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')

# ===============================
# Step 3: Remove invalid billing entries
# ===============================
# Billing Amounts must be non-negative
df = df[df['Billing Amount'] >= 0].copy()

# ===============================
# Step 4: Exploratory Data Analysis (EDA)
# ===============================
# Plot distribution of Age
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')

# Plot distribution of Billing Amount
plt.subplot(1, 2, 2)
sns.histplot(df['Billing Amount'], bins=30, kde=True)
plt.title('Billing Amount Distribution')
plt.tight_layout()
plt.show()

# ===============================
# Step 5: Categorical Feature Distributions
# ===============================
categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 
                    'Insurance Provider', 'Admission Type', 
                    'Medication', 'Test Results']

# Create count plots for each categorical column
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, ax=axes[idx], order=df[col].value_counts().index)
    axes[idx].set_title(f'{col} Distribution')
    axes[idx].tick_params(axis='x', rotation=45)

# Remove any unused subplots
for j in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ===============================
# Step 6: Save cleaned data
# ===============================
# Export cleaned version for later preprocessing and modeling
df.to_csv("cleaned_healthcare_dataset.csv", index=False)

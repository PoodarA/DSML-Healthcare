import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_healthcare_dataset.csv")

print(df.info())
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

print("\nDescriptive Stats:")
print(df[['Length of Stay']].describe())

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(df['Length of Stay'].dropna(), bins=30, kde=True)
plt.title("Length of Stay Distribution")
plt.xlabel("Days")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Gender')
plt.title("Gender Distribution")
plt.show()

top_diagnoses = df['Medical Condition'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_diagnoses.index, y=top_diagnoses.values)
plt.xticks(rotation=45)
plt.title("Top 10 Medical Conditions")
plt.ylabel("Frequency")
plt.show()


negatives = df[df['Billing Amount'] < 0]
print(f"Number of records with negative billing: {len(negatives)}")

plt.figure(figsize=(10, 6))
sns.histplot(df['Billing Amount'], bins=40, kde=True)
plt.title("Distribution of Billing Amount")
plt.xlabel("Billing Amount ($)")
plt.ylabel("Number of Patients")
plt.show()


df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Count by group
sns.countplot(data=df, x='age_group')
plt.title("Patients by Age Group")
plt.show()

import pandas as pd

print("Loading Titanic Dataset...")

df = pd.read_csv("Titanic.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values

if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

if "Cabin" in df.columns:
    df["Cabin"] = df["Cabin"].fillna("Unknown")

# Remove duplicates

df.drop_duplicates(inplace=True)

# Rename columns

df.rename(columns={
    "Pclass": "Passenger_Class",
    "SibSp": "Siblings_Spouses",
    "Parch": "Parents_Children"
}, inplace=True)

print("\nDataset After Cleaning:")
print(df.head())

df.to_csv(
    "Titanic_Cleaned.csv",
    index=False
)

print("\nTitanic_Cleaned.csv Saved Successfully")
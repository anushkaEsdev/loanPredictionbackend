import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


# Load dataset (UPDATED FILE NAME)
df = pd.read_csv("Loan-Approval-Prediction.csv")

# Drop Loan_ID column if it exists
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Handle missing values
df.fillna({
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Credit_History": 1,
    "Property_Area": "Semiurban",
    "LoanAmount": df["LoanAmount"].median(),
    "Loan_Amount_Term": df["Loan_Amount_Term"].median(),
    "ApplicantIncome": df["ApplicantIncome"].median(),
    "CoapplicantIncome": df["CoapplicantIncome"].median()
}, inplace=True)

# Convert categorical values to numerical
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Property_Area"] = df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})  # Fix Y/N issue
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)  # Fix 3+ issue

# Convert all columns to float
df = df.astype(float)

# Splitting data
X = df.drop(columns=["Loan_Status"])  # Features
y = df["Loan_Status"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
import pickle

# After training the model, save it as a pickle file
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as loan_model.pkl")

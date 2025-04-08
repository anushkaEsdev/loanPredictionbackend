import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv('Loan-Approval-Prediction.csv')

# Handle missing values
df['Dependents'] = df['Dependents'].fillna(0)
df['Loan_Amount'] = df['Loan_Amount'].fillna(df['Loan_Amount'].mean())
df['Loan_Term'] = df['Loan_Term'].fillna(360)

# Convert categorical variables to numerical
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Marital_Status'] = df['Marital_Status'].map({'Married': 1, 'Single': 0})
df['Education_Level'] = df['Education_Level'].map({'Graduate': 1, 'Not Graduate': 0})
df['Employment_Status'] = df['Employment_Status'].map({'Salaried': 1, 'Self Employed': 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 0, 'Semiurban': 1, 'Rural': 2})
df['Credit_History'] = df['Credit_History'].map({'Good': 1, 'Bad': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Feature selection
features = ['Gender', 'Marital_Status', 'Dependents', 'Education_Level', 
           'Employment_Status', 'Monthly_Income', 'Co_Applicant_Income', 
           'Loan_Amount', 'Loan_Term', 'Credit_History', 'Property_Area']
X = df[features]
y = df['Loan_Status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print model performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'loan_approval_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Function to predict loan approval
def predict_loan_approval(data):
    # Convert input data to match training data format
    input_data = pd.DataFrame([data])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return {
        'approved': bool(prediction),
        'probability': float(probability)
    } 
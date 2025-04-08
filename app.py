from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the model and scaler
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route("/", methods=["GET"])
def home():
    """ Health Check Endpoint """
    return jsonify({"message": "✅ Loan Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """ API endpoint to predict loan approval """
    try:
        data = request.get_json()
        
        # Extract features in the correct order
        input_data = {
            'Gender': 1 if data['gender'] == 'Male' else 0,
            'Marital_Status': 1 if data['marital_status'] == 'Married' else 0,
            'Dependents': int(data['dependents']),
            'Education_Level': 1 if data['education'] == 'Graduate' else 0,
            'Employment_Status': 1 if data['employment_status'] == 'Salaried' else 0,
            'Monthly_Income': float(data['monthly_income']),
            'Co_Applicant_Income': float(data['co_applicant_income']),
            'Loan_Amount': float(data['loan_amount']),
            'Loan_Term': int(data['loan_term']),
            'Credit_History': 1 if data['credit_history'] == 'Good' else 0,
            'Property_Area': {'Urban': 0, 'Semiurban': 1, 'Rural': 2}[data['property_area']]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return jsonify({
            'approved': bool(prediction),
            'probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

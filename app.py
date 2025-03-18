from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'loan_model.pkl')
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route("/", methods=["GET"])
def home():
    """ Health Check Endpoint """
    return jsonify({"message": "✅ Loan Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """ API endpoint to predict loan approval """
    try:
        data = request.get_json()

        # Ensure all required fields are present
        required_fields = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Convert input into the required format
        input_features = np.array([data[field] for field in required_fields]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)
        result = "Approved" if prediction[0] == 1 else "Rejected"

        return jsonify({"loan_approval": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

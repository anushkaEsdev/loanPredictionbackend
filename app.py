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

        # Preprocess input data
        processed_data = {
            "Gender": 1 if data["Gender"].lower() == "male" else 0,
            "Married": 1 if data["Married"].lower() == "yes" else 0,
            "Dependents": int(data["Dependents"]),
            "Education": 1 if data["Education"].lower() == "graduate" else 0,
            "Self_Employed": 1 if data["Self_Employed"].lower() == "yes" else 0,
            "ApplicantIncome": float(data["ApplicantIncome"]),
            "CoapplicantIncome": float(data["CoapplicantIncome"]),
            "LoanAmount": float(data["LoanAmount"]),
            "Loan_Amount_Term": float(data["Loan_Amount_Term"]),
            "Credit_History": float(data["Credit_History"]),
            "Property_Area": {"urban": 2, "semiurban": 1, "rural": 0}.get(data["Property_Area"].lower(), 1)
        }

        # Convert input into the required format
        input_features = np.array([processed_data[field] for field in required_fields]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features)[0][1]  # Probability of approval
        
        result = "Approved" if prediction[0] == 1 else "Rejected"
        
        return jsonify({
            "loan_approval": result,
            "approval_probability": float(probability),
            "input_features": processed_data
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

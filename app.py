from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = "C:/Users/gadha/OneDrive/Desktop/final project/churn_model.pkl"
FEATURES_PATH = "C:/Users/gadha/OneDrive/Desktop/final project/feature_names.json"

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")

    # Load feature names separately
    with open(FEATURES_PATH, "r") as f:
        feature_order = json.load(f)
        print("✅ Feature names loaded successfully!")

else:
    print("❌ ERROR: Model or feature_names.json file not found!")
    model = None
    feature_order = None

@app.route('/')
def home():
    return "🔥 Churn Prediction API is Running! Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or feature_order is None:
        return jsonify({"error": "Model or feature names not loaded correctly."})

    try:
        # Get JSON input
        data = request.get_json()
        print("Received Data:", data)  # Debugging step

        if not data:
            return jsonify({"error": "No data received!"})

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # One-hot encode categorical values
        categorical_columns = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]

        df = pd.get_dummies(df)  # Convert categorical columns to one-hot encoding

        # Ensure all model features are present (missing ones should be filled with 0)
        df = df.reindex(columns=feature_order, fill_value=0)

        print("Final DataFrame for Prediction:", df)  # Debugging step

        # Make prediction
        prediction = model.predict(df)
        print("Prediction Result:", prediction)  # Debugging step

        # Return response
        return jsonify({"churn_prediction": int(prediction[0])})

    except Exception as e:
        print("Error:", str(e))  # Debugging step
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

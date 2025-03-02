from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model and feature names
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model1.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.json")

model, feature_order = None, None

# Check if model and feature files exist
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {str(e)}")
else:
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}")

if os.path.exists(FEATURES_PATH):
    try:
        with open(FEATURES_PATH, "r") as f:
            feature_order = json.load(f)
        print("✅ Feature names loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR: Failed to load feature names: {str(e)}")
else:
    print(f"❌ ERROR: Feature file not found at {FEATURES_PATH}")

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

        # Define categorical columns for one-hot encoding
        categorical_columns = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]

        # One-hot encode categorical values
        df = pd.get_dummies(df)

        # Ensure all expected features are present, fill missing ones with 0
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

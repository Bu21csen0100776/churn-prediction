{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "faf7650b-b733-4c1e-a905-ef2db7ce6d14",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "from flask_cors import CORS\n",
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "# Initialize Flask app\n",
    "app = Flask(__name__)\n",
    "CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)\n",
    "\n",
    "# Load the trained churn prediction model\n",
    "model = joblib.load(\"churn_model.pkl\")  # Ensure this file exists\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return \"Churn Prediction API is running!\"\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    try:\n",
    "        # Get JSON data from the request\n",
    "        data = request.get_json()\n",
    "\n",
    "        # Convert JSON to DataFrame (ensure the keys match the model's training features)\n",
    "        df = pd.DataFrame([data])  \n",
    "\n",
    "        # Make prediction\n",
    "        prediction = model.predict(df)\n",
    "\n",
    "        # Return response\n",
    "        return jsonify({\"churn_prediction\": int(prediction[0])})\n",
    "\n",
    "    except Exception as e:\n",
    "        return jsonify({\"error\": str(e)})\n",
    "\n",
    "# Run Flask app\n",
    "if __name__ == '__main__':\n",
    "    app.run(host='0.0.0.0', port=5000, debug=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c8df18d4-50ef-4a54-b75c-a82ff37befc9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

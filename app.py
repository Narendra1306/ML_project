"""
Backend Flask Application
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

# Load trained model
try:
    with open("SLR_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print("Error loading model:", e)

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        inputs = [float(x) for x in request.form.values()]

        # Convert to numpy array for model
        final_input = [np.array(inputs)]

        # Make prediction
        result = model.predict(final_input)[0]

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# Run locally
if __name__ == "__main__":
    app.run(debug=True)
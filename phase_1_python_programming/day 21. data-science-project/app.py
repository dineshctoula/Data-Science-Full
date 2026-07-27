import pickle
from flask import Flask, jsonify, render_template, request
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  # Ensure scaler was saved during training

# =========================
# WEB INTERFACE (HTML UI)
# =========================

@app.route("/")
def home():
    """Renders the HTML form frontend."""
    return render_template("index.html")

@app.route("/predict-ui", methods=["POST"])
def predict_ui():
    """Handles HTML form submissions from index.html."""
    try:
        # Extract numerical features from HTML form inputs
        features = [float(x) for x in request.form.values()]
        raw_features = np.array(features).reshape(1, -1)

        # Scale features before feeding to the model
        scaled_features = scaler.transform(raw_features)
        prediction = model.predict(scaled_features)

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Value: ${prediction[0]:,.2f}"
        )

    except ValueError:
        return render_template(
            "index.html",
            prediction_text="Error: Please enter valid numerical values for all fields."
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Prediction Error: {str(e)}"
        )

# =========================
# REST API (JSON ENDPOINT)
# =========================

@app.route("/api/predict", methods=["POST"])
def predict_api():
    """Handles raw JSON API requests (e.g., via Postman, React, or Axios)."""
    try:
        data = request.get_json(force=True)
        
        if "features" not in data:
            return jsonify({"status": "error", "message": "Missing 'features' key in JSON body."}), 400

        raw_features = np.array(data["features"]).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(raw_features)
        prediction = model.predict(scaled_features)

        return jsonify({
            "status": "success",
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# =========================
# SERVER EXECUTION
# =========================

if __name__ == "__main__":
    app.run(debug=True)
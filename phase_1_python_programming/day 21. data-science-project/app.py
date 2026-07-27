# =========================
# COMMIT 9: FLASK API
# =========================

from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return "✅ ML Model API is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        features = np.array(data).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": prediction.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Run server
if __name__ == "__main__":
    app.run(debug=True)





from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

# Home page (UI)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction from form
@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Value: {prediction[0]:.2f}"
        )

    except:
        return render_template(
            "index.html",
            prediction_text="Error in input!"
        )

if __name__ == "__main__":
    app.run(debug=True)
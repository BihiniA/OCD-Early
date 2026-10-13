from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# ============================================================
# OCD-Early Flask API
# Connects the trained ML model to the mobile frontend
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "best_ocd_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Place the trained model file in the same folder as this app."
    )

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"Label encoder file not found: {ENCODER_PATH}. "
        "Place the label encoder file in the same folder as this app."
    )

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ============================================================
# 2. MODEL INPUT FEATURES
# ============================================================

child_columns = [f"Child_Q{i}" for i in range(1, 19)]
parent_columns = [f"Parent_Q{i}" for i in range(1, 19)]

feature_columns = ["Age"] + child_columns + parent_columns

# ============================================================
# 3. HOME / API INFORMATION
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "OCD-Early Flask API is running",
        "version": "1.0",
        "prediction_endpoint": "/api/predict",
        "health_endpoint": "/api/health",
        "method": "POST",
        "required_features": len(feature_columns),
        "features": feature_columns
    })

# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None
    })

# ============================================================
# 5. PREDICTION ENDPOINT
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Read JSON request from the mobile application.
        data = request.get_json(silent=True)

        if data is None:
            return jsonify({
                "success": False,
                "error": "Request body must contain valid JSON data."
            }), 400

        # Check that all 37 model inputs were supplied.
        missing_fields = [
            column for column in feature_columns
            if column not in data
        ]

        if missing_fields:
            return jsonify({
                "success": False,
                "error": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400

        # Convert model inputs to numeric values.
        input_data = {}

        for column in feature_columns:
            try:
                input_data[column] = float(data[column])
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": f"{column} must be a numeric value."
                }), 400

        # Keep exactly the same feature order used during training.
        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        # Generate prediction.
        prediction_encoded = model.predict(input_df)
        predicted_class = int(prediction_encoded[0])

        # Decode the numeric class into the original risk label.
        predicted_risk = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        # Random Forest supports predict_proba(). If the selected model
        # does not, the API simply omits probabilities.
        probabilities = None

        if hasattr(model, "predict_proba"):
            probability_values = model.predict_proba(input_df)[0]
            probabilities = {
                str(class_name): round(float(probability), 4)
                for class_name, probability in zip(
                    label_encoder.classes_,
                    probability_values
                )
            }

        response = {
            "success": True,
            "prediction": {
                "risk_level": str(predicted_risk),
                "encoded_class": predicted_class
            },
            "message": "Prediction generated successfully."
        }

        if probabilities is not None:
            response["prediction"]["probabilities"] = probabilities

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "An error occurred while making the prediction.",
            "details": str(e)
        }), 500

# ============================================================
# 6. RUN FLASK SERVER
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

import os
import json
import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory
)

from flask_cors import CORS
from preprocessing import preprocess_image
from database import save_prediction


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "..",
    "frontend"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plant_growth_model.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.json"
)


# --------------------------------------------------
# TensorFlow CPU settings
# --------------------------------------------------

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# --------------------------------------------------
# Flask
# --------------------------------------------------

app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Load AI Model
# --------------------------------------------------

print("Loading AI model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

print("AI model loaded!")
print("Classes:", class_names)


# --------------------------------------------------
# Warm up the model
# --------------------------------------------------

print("Warming up AI model...")

dummy_image = np.zeros(
    (1, 224, 224, 3),
    dtype=np.float32
)

model(
    dummy_image,
    training=False
)

print("AI model ready!")


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    try:

        print("Prediction request received")

        # ------------------------------------------
        # Preprocess image
        # ------------------------------------------

        image = preprocess_image(file)

        print(
            "Image preprocessed:",
            image.shape
        )

        # ------------------------------------------
        # Model prediction
        # ------------------------------------------

        predictions = model(
            image,
            training=False
        ).numpy()[0]

        print("Prediction completed")

        predicted_index = int(
            np.argmax(predictions)
        )

        stage = class_names[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index] * 100
        )

        print(
            f"Prediction: {stage} "
            f"({confidence:.2f}%)"
        )

        # ------------------------------------------
        # Save prediction to MongoDB
        # ------------------------------------------

        try:

            save_prediction(
                file.filename,
                stage,
                confidence
            )

            print("Prediction saved to MongoDB")

        except Exception as database_error:

            print(
                "Prediction was not saved to MongoDB:",
                database_error
            )

        # ------------------------------------------
        # Return JSON response
        # ------------------------------------------

        return jsonify({
            "stage": stage,
            "confidence": round(
                confidence,
                2
            )
        }), 200

    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return jsonify({
            "error": "Prediction failed",
            "details": str(error)
        }), 500


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# --------------------------------------------------
# Run locally
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
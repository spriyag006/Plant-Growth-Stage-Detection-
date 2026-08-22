import os
import json

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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__)

CORS(app)


# ---------------------------------------
# Paths
# ---------------------------------------

MODEL_PATH = os.path.join(
    "model",
    "plant_growth_model.keras"
)

CLASS_PATH = os.path.join(
    "model",
    "class_names.json"
)


# ---------------------------------------
# Load AI Model
# ---------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ---------------------------------------
# Load Class Names
# ---------------------------------------

with open(
    CLASS_PATH,
    "r"
) as file:

    class_names = json.load(
        file
    )


print("AI model loaded!")

print(
    "Classes:",
    class_names
)


# ---------------------------------------
# Prediction API
# ---------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
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

        # Preprocess

        image = preprocess_image(
            file
        )


        # AI Prediction

        predictions = model.predict(
            image,
            verbose=0
        )[0]


        predicted_index = int(
            tf.argmax(predictions)
        )


        stage = class_names[
            predicted_index
        ]


        confidence = float(
            predictions[
                predicted_index
            ] * 100
        )


        # Keep inference available even when the optional database is offline.
        try:
            save_prediction(
                file.filename,
                stage,
                confidence
            )
        except Exception as database_error:
            print("Prediction was not saved to MongoDB:", database_error)


        return jsonify({

            "stage": stage,

            "confidence": round(
                confidence,
                2
            )

        })


    except Exception as error:

        print(error)

        return jsonify({

            "error":
            "Prediction failed"

        }), 500


# ---------------------------------------
# Home
# ---------------------------------------

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ---------------------------------------
# Run Flask
# ---------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )
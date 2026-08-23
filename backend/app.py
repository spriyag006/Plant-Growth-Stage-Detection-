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

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

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


       

        import time
        start = time.time()
        predictions = model.predict(image, verbose=0)[0]
        print("Inference took:", time.time() - start, "seconds")


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

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


if __name__ == "__main__":
   
    port = int(os.environ.get("PORT",5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
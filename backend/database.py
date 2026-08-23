import os

from pymongo import MongoClient
from dotenv import load_dotenv


load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError(
        "MONGO_URI is not set in the .env file"
    )

client = MongoClient(MONGO_URI)
print(client.list_database_names())
db = client["plant_growth_db"]

predictions_collection = db["predictions"]


def save_prediction(
    image_name,
    stage,
    confidence
):

    document = {
        "image_name": image_name,
        "predicted_stage": stage,
        "confidence": confidence
    }

    predictions_collection.insert_one(
        document
    )

    print("Prediction saved to MongoDB")
import os

from pymongo import MongoClient
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Get MongoDB connection string
MONGO_URI = os.getenv("MONGO_URI")


# Check MongoDB URI
if not MONGO_URI:
    raise ValueError(
        "MONGO_URI is not set in the .env file"
    )


# Connect to MongoDB
client = MongoClient(MONGO_URI)


# Select database
db = client["plant_growth_db"]


# Select collection
predictions_collection = db["predictions"]


# Save prediction
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
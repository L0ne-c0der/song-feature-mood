from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

EXPECTED_FEATURES = 9

# Define class labels (adjust as needed based on your model's classes)
CLASS_LABELS = ["sad", "happy", "energitic", "calm"]

# Load artifacts (must be in same folder)
scaler = joblib.load("scaler.pkl")
model = tf.keras.models.load_model("model.h5")

# Create app
app = FastAPI()

# Input schema
class InputData(BaseModel):
    danceability: float
    energy: float
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float

@app.post("/predict")
def predict(item: InputData):
    # Extract features in the correct order
    features = [
        item.danceability,
        item.energy,
        item.loudness,
        item.speechiness,
        item.acousticness,
        item.instrumentalness,
        item.liveness,
        item.valence,
        item.tempo
    ]
    
    x = np.array(features, dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)
    preds = model.predict(x_scaled)
    
    # Create predictions dictionary with labels
    predictions = {CLASS_LABELS[i]: float(preds[0][i]) for i in range(len(CLASS_LABELS))}
    predicted_class = CLASS_LABELS[int(np.argmax(preds[0]))]
    
    return {
        "predictions": predictions,
        "predicted_class": predicted_class
    }
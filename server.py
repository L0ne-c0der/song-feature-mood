from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

EXPECTED_FEATURES = 9

# Load artifacts (must be in same folder)
scaler = joblib.load("scaler.pkl")
model = tf.keras.models.load_model("model.h5")

# Create app
app = FastAPI()

# Input schema
class InputData(BaseModel):
    data: list[float]
    # order:-
    # danceability	
    # energy	
    # loudness	
    # speechiness	
    # acousticness	
    # instrumentalness	
    # liveness	
    # valence	
    # tempo	

    # Ex: "data":[0.598,0.705,-5.525,0.497,0.0362,0.497,0.000128,0.0878,75.023]

@app.post("/predict")
def predict(item: InputData):
    if len(item.data) != EXPECTED_FEATURES:
        return {
            "error": f"Expected {EXPECTED_FEATURES} features, got {len(item.data)}"
        }

    x = np.array(item.data, dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)
    preds = model.predict(x_scaled)

    return {
        "softmax": preds[0].tolist(),
        "predicted_class": int(np.argmax(preds[0]))
    }

#command via cURL:
# curl -X POST http://127.0.0.1:8000/predict \
# -H "Content-Type: application/json" \
# -d '{"data":[0.598,0.705,-5.525,0.497,0.0362,0.497,0.000128,0.0878,75.023]}'

# sample output:
 #{"softmax":[0.002706807805225253,0.19403111934661865,0.8031859993934631,7.602207188028842e-05],"predicted_class":2}
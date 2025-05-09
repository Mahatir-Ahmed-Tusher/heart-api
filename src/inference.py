import pickle
import numpy as np
import aiohttp
import json
import os
from .utils import format_symptoms_for_report

# Load the trained model
# Compute the path to model.pkl relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of inference.py (src/)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")  # Go up to Heart/, then into models/
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Define the expected feature order
FEATURES = [
    "pain_arms_jaw_back",
    "age",
    "cold_sweats_nausea",
    "chest_pain",
    "fatigue",
    "dizziness",
    "swelling",
    "shortness_of_breath",
    "palpitations",
    "sedentary_lifestyle",
]

async def predict_heart_risk_and_generate_report(input_data: dict, mistral_api_key: str):
    # Extract features in the correct order
    input_features = [float(input_data[feature]) for feature in FEATURES]
    final_features = np.array([input_features])

    # Make prediction
    prediction = model.predict(final_features)
    prediction_text = "At Risk" if prediction[0] == 1 else "Not at Risk"

    # Format symptoms for the report
    symptoms_text = format_symptoms_for_report(input_data)

    # Generate prompt for Mistral AI
    prompt = (
        f"The user has provided the following symptoms:\n{symptoms_text}\n"
        f"Based on these symptoms, a machine learning model predicts that the user is '{prediction_text}' for heart disease.\n"
        f"If the prediction is 'At Risk', explain to the user what can be done to stay healthy and keep the heart healthy. If the prediction is 'At Risk' then explain why they are at risk based on the symptoms they provided. If not at risk, explain why they are not at risk based on the symptoms they provided. "
        f"Suggest a diet specific to South Asia and explain how to recover from or manage the specific symptoms they answered 'Yes' to. "
        f"If the prediction is 'Not at Risk', for the symptoms where the user answered 'Yes', explain how to stay well and keep the heart healthy. "
        f"Additionally, provide general heart health advice."
    )

    # Call Mistral AI API
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {mistral_api_key}",
            },
            data=json.dumps({
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3000,
                "temperature": 0.7,
                "top_p": 0.9,
            }),
        ) as response:
            if response.status != 200:
                raise Exception(f"Mistral AI API error: {await response.text()}")
            result = await response.json()
            report = result["choices"][0]["message"]["content"]

    return {
        "prediction": prediction_text,
        "report": report,
    }
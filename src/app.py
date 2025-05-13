import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from .inference import predict_heart_risk_and_generate_report
import uvicorn  # Add this import

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env")

# Initialize FastAPI app
app = FastAPI(title="Heart Risk Assessment API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://earlymed.vercel.app"],  # Update later if frontend is deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for user input
class HeartRiskInput(BaseModel):
    pain_arms_jaw_back: int
    age: float
    cold_sweats_nausea: int
    chest_pain: int
    fatigue: int
    dizziness: int
    swelling: int
    shortness_of_breath: int
    palpitations: int
    sedentary_lifestyle: int

# API endpoint for prediction and report generation
@app.post("/api/heart-risk")
async def heart_risk(input_data: HeartRiskInput):
    try:
        # Convert input data to dictionary
        input_dict = input_data.dict()
        
        # Get prediction and report
        result = await predict_heart_risk_and_generate_report(input_dict, MISTRAL_API_KEY)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Add main block for Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT not set
    uvicorn.run(app, host="0.0.0.0", port=port)

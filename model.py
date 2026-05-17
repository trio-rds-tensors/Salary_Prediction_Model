from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Salary_Prediction_Model import Salary
import uvicorn

# Initialize the model at startup
model = Salary()

app = FastAPI(title="Salary Predictor API")

# Allow CORS so the frontend app can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    education: str
    job_title: str
    experience_years: float

class PredictionResponse(BaseModel):
    predicted_salary: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Predict the salary using the loaded model
    prediction = model.predict([request.education, request.job_title, request.experience_years])
    
    # Extract the scalar predicted value (handles different numpy array shapes)
    try:
        predicted_value = float(prediction[0][0])
    except (IndexError, TypeError):
        predicted_value = float(prediction[0])
        
    return {"predicted_salary": predicted_value}

if __name__ == "__main__":
    # You can run this file directly with `python model.py`
    uvicorn.run("model:app", host="0.0.0.0", port=8000, reload=True)
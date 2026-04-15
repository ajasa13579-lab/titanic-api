from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Titanic Survival Prediction API")

# Load the "Brain"
MODEL_PATH = "models/titanic_model.joblib"
IMPUTER_PATH = "models/titanic_imputer.joblib"

model = None
imputer = None
if os.path.exists(MODEL_PATH) and os.path.exists(IMPUTER_PATH):
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
elif os.path.exists("titanic_model.joblib"):
    model = joblib.load("titanic_model.joblib")
    imputer = joblib.load("titanic_imputer.joblib")

class PassengerInfo(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Titanic Survival Prediction API! This is the Waiter."}

@app.post("/predict")
def predict_survival(passenger: PassengerInfo):
    if model is None or imputer is None:
        raise HTTPException(status_code=500, detail="The 'Brain' (model) is not loaded yet.")
    
    sex_encoded = 0 if passenger.Sex.lower() == 'male' else 1
    input_data = [[passenger.Pclass, sex_encoded, passenger.Age, passenger.Fare]]
    input_data_imputed = imputer.transform(input_data)
    
    prediction = model.predict(input_data_imputed)[0]
    return {
        "survived": bool(prediction)
    }

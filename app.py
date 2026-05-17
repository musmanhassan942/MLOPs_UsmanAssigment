from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Literal
# Create FastAPI app
app = FastAPI()

# Load model and training columns
model = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")

# Pydantic model
class PatientData(BaseModel):
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: Literal["M", "F"]

# Health check endpoint
@app.get("/")
def home():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):

    # Convert input into dictionary
    input_data = data.model_dump()

    # Convert into DataFrame
    df = pd.DataFrame([input_data])

    # One-hot encoding
    df = pd.get_dummies(df, columns=["gender"])

    # Match training columns
    df = df.reindex(columns=training_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(df)

    # Return prediction
    return {
        "prediction": prediction[0]
    }
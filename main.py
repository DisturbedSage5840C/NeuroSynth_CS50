from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")


class PatientData(BaseModel):
    age: float
    educ: float
    ses: float
    mmse: float
    cdr: float
    etiv: float
    nwbv: float
    asf: float


@app.get("/")
def root():
    return {"status": "NeuroSynth is running"}


@app.post("/predict")
def predict(data: PatientData):
    features = pd.DataFrame(
        [[
            data.age,
            data.educ,
            data.ses,
            data.mmse,
            data.cdr,
            data.etiv,
            data.nwbv,
            data.asf,
        ]],
        columns=["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"],
    )
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()
    label = le.inverse_transform([prediction])[0]
    return {
        "prediction": label,
        "confidence": round(float(probability) * 100, 2),
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import TextRequest, PredictionResponse
from src.inference.engine import PersonalityEngine

app = FastAPI(
    title="Personality Trait Prediction API",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
engine = PersonalityEngine()


@app.get("/")
def root():
    return {"message": "Personality Prediction API is running."}


@app.post("/predict")
def predict(request: TextRequest):
    return engine.predict_all_models(request.text)


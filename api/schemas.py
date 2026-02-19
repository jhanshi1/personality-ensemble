from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    OPN: float
    CON: float
    EXT: float
    AGR: float
    NEU: float

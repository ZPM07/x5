from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.predictor import Predictor

app = FastAPI(
    title="NER Inference API",
    description="API для аннотации сущностей с помощью NER",
    version="1.0.0"
)

predictor = Predictor()


class PredictRequest(BaseModel):
    input: str


@app.post("/api/predict")
def predict(request: PredictRequest) -> List[Dict[str, Any]]:
    text = request.input.strip()

    if not text:
        return []

    try:
        annotations = predictor.predict(text)

        result = [
            {"start_index": start, "end_index": end, "entity": label}
            for (start, end, label) in annotations
        ]

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
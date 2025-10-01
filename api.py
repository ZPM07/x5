from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

from pipeline import process_example

app = FastAPI(
    title="NER Inference API",
    description="API для аннотации сущностей через RuBERT + XGBoost",
    version="1.0.0"
)

class TextRequest(BaseModel):
    texts: List[str]

@app.post("/predict")
def predict(request: TextRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Список текстов пуст.")

    try:
        df = pd.DataFrame({"sample": request.texts})
        result_df = process_example(df)

        return {
            "results": [
                {
                    "text": row["sample"],
                    "annotations": row["annotation"]
                } for _, row in result_df.iterrows()
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

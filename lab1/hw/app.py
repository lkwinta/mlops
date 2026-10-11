from fastapi import FastAPI
from api.models.sentiment import PredictRequest, PredictResponse
from inference import SentimentAnalysis

app = FastAPI()
sentiment_analysis = SentimentAnalysis()

@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction=sentiment_analysis.predict(request.text))

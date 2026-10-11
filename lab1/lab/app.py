from fastapi import FastAPI
import inference
from api.models.iris import PredictRequest, PredictResponse

app = FastAPI()
iris_model = inference.load_model()

@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    prediction = inference.predict(iris_model, request.model_dump())
    return PredictResponse(prediction=prediction)

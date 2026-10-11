import pytest
from fastapi.testclient import TestClient
from joblib import load

from api.models.sentiment import PredictResponse
from app import app
from inference import CLASSIFIER_PATH


client = TestClient(app)


def test_classifier_is_loaded_from_provided_file_without_errors():
    classifier = load(CLASSIFIER_PATH)

    assert callable(classifier.predict)


@pytest.mark.parametrize(
    ("text", "expected_prediction"),
    [
        ("I love this product!", "positive"),
        ("The package arrived on Tuesday.", "neutral"),
        ("This is absolutely terrible.", "negative"),
    ],
)
def test_model_inference_works_for_sample_strings(
    text: str,
    expected_prediction: str,
):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    assert response.json() == {"prediction": expected_prediction}


def test_predict_endpoint_returns_a_valid_json_response():
    response = client.post("/predict", json={"text": "I love this product!"})
    response_body = response.json()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert PredictResponse.model_validate(response_body).prediction == "positive"


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"text": ""},
        {"text": 123},
        {},
    ],
)
def test_invalid_input_returns_json_with_error_explanation(invalid_payload):
    response = client.post("/predict", json=invalid_payload)
    response_body = response.json()

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response_body.get("detail"), list)
    assert response_body["detail"]
    assert all(error.get("msg") for error in response_body["detail"])

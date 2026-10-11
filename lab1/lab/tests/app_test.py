from fastapi import FastAPI
from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


def test_welcome_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML API"}


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_setosa():
    request_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    assert response.json() == {"prediction": "setosa"}

def test_predict_versicolor():
    request_data = {
        "sepal_length": 6.0,
        "sepal_width": 2.9,
        "petal_length": 4.5,
        "petal_width": 1.5
    }
    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    assert response.json() == {"prediction": "versicolor"}

def test_predict_virginica():
    request_data = {
        "sepal_length": 6.3,
        "sepal_width": 3.3,
        "petal_length": 6.0,
        "petal_width": 2.5
    }
    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    assert response.json() == {"prediction": "virginica"}

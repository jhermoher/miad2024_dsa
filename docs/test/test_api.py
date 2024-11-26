# tests/test_api.py
import sys
import os
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint(client, sample_news_input):
    response = client.post("/predict", json=sample_news_input)
    assert response.status_code == 200
    assert "category" in response.json()
    assert "sentiment" in response.json()

def test_invalid_input(client):
    response = client.post("/predict", json={"headline ": ""})
    assert response.status_code != 200

# tests/conftest.py
import sys
import os
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_news_input():
    return {
        "header": "Tech Giant Announces New AI Features",
        "summary": "Major technology company reveals innovative artificial intelligence capabilities",
        "date": "2024-01-01"
    }
# tests/test_pipeline.py
import sys
import os
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pytest
from app.pipeline import NewsPipeline

@pytest.fixture
def pipeline():
    return NewsPipeline()

def test_preprocessing(pipeline):
    text = "Tech Giant Announces New AI Features"
    result = pipeline.preprocess_text(text)
    assert "topic" in result
    assert "sentiment" in result
    assert "pos_tags" in result

def test_sentiment_analysis(pipeline):
    text = "Great new features announced"
    result = pipeline.analyze_sentiment(text)
    assert "sentiment" in result
    assert "compound" in result

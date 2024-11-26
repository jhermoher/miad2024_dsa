# app/schemas.py
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List

class NewsInput(BaseModel):
    headline: str
    short_description: str
    date: Optional[str] = None

class CategoryPrediction(BaseModel):
    predicted: str
    confidence: float

class SentimentScores(BaseModel):
    compound: float
    positive: float
    neutral: float
    negative: float

class SentimentResult(BaseModel):
    label: str
    scores: SentimentScores

class POSFeatures(BaseModel):
    adj_count: int
    verb_count: int
    noun_count: int
    proper_noun_count: int
    adv_count: int
    quote_count: int

class NewsPatterns(BaseModel):
    named_entities: List[str]
    action_phrases: List[str]
    descriptive_phrases: List[str]
    quotes: List[str]
    temporal_expressions: List[str]
    location_expressions: List[str]

class ProcessedData(BaseModel):
    date: str
    processed_text: str
    pos_features: POSFeatures
    news_patterns: NewsPatterns

class PredictionResponse(BaseModel):
    success: bool
    category: Optional[CategoryPrediction] = None
    sentiment: Optional[SentimentResult] = None
    processed_data: Optional[ProcessedData] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
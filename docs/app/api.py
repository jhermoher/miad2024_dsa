# app/api.py
# app/api.py
import logging
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import settings
from .schemas import (
    NewsInput, 
    PredictionResponse, 
    HealthResponse, 
    CategoryPrediction,
    SentimentResult,
    ProcessedData
)
from .pipeline import NewsPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    News Classification and Sentiment Analysis API
    
    This API provides endpoints for:
    - Categorizing news articles
    - Analyzing sentiment
    - Extracting linguistic features
    - Health monitoring
    """
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
try:
    pipeline = NewsPipeline(model_dir=settings.MODEL_PATH)
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {str(e)}")
    raise

# Custom exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "News Classification API",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify API status"""
    try:
        # Perform basic model check
        test_text = "Test headline"
        pipeline.preprocess_text(test_text)
        
        return HealthResponse(
            status="healthy",
            version=settings.APP_VERSION
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version=settings.APP_VERSION
        )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(news: NewsInput):
    """
    Predict category and sentiment for a news article
    
    Parameters:
    - headline: News article headline
    - short_description: Brief description or summary of the article
    - date: Optional publication date (YYYY-MM-DD format)
    
    Returns:
    - Category prediction with confidence score
    - Sentiment analysis results
    - Processed text features and patterns
    """
    try:
        # Input validation
        if not news.headline or not news.headline.strip():
            raise HTTPException(
                status_code=400,
                detail="Headline cannot be empty"
            )
        if not news.short_description or not news.short_description.strip():
            raise HTTPException(
                status_code=400,
                detail="Short description cannot be empty"
            )

        # Log request
        logger.info(f"Processing prediction request for headline: {news.headline[:50]}...")
        
        # Make prediction
        result = pipeline.predict(
            headline=news.headline,
            short_description=news.short_description,
            date=news.date
        )
        
        if not result["success"]:
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"Prediction failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.info("Prediction completed successfully")
        
        # Validate response
        try:
            return PredictionResponse(
                success=True,
                category=CategoryPrediction(
                    predicted=result["category"]["predicted"],
                    confidence=result["category"]["confidence"]
                ),
                sentiment=SentimentResult(
                    label=result["sentiment"]["label"],
                    scores=result["sentiment"]["scores"]
                ),
                processed_data=ProcessedData(
                    date=result["processed_data"]["date"],
                    processed_text=result["processed_data"]["processed_text"],
                    pos_features=result["processed_data"]["pos_features"],
                    news_patterns=result["processed_data"]["news_patterns"]
                )
            )
        except ValidationError as e:
            logger.error(f"Response validation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error formatting prediction response"
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/models/info", tags=["Models"])
async def get_models_info():
    """Get information about loaded models and their configurations"""
    try:
        return {
            "models": {
                "vectorizers": {
                    "count_vectorizer": str(pipeline.count_vectorizer),
                    "tfidf_vectorizer": str(pipeline.tfidf_vectorizer)
                },
                "dimension_reduction": {
                    "svd": str(pipeline.svd),
                    "lda": str(pipeline.lda)
                },
                "classifier": str(pipeline.rf),
                "scaler": str(pipeline.scaler)
            },
            "nltk_resources": [
                "punkt_tab",
                "averaged_perceptron_tagger_eng",
                "maxent_ne_chunker_tab",
                "words",
                "stopwords",
                "wordnet"
            ],
            "sentiment_analyzer": "VADER"
        }
    except Exception as e:
        logger.error(f"Error retrieving models info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving models information"
        )

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="News Classification and Sentiment Analysis API",
        routes=app.routes,
    )
    
    # Add security schemes if needed
    # openapi_schema["components"]["securitySchemes"] = {...}
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

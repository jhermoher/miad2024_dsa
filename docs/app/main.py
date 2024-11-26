# app/main.py
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import NewsInput, PredictionResponse, HealthResponse
from .pipeline import NewsPipeline
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="News Classification and Sentiment Analysis API"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {"message": "News Classification API", "version": settings.APP_VERSION}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(news: NewsInput):
    """
    Predict category and sentiment for a news article
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
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

def start_server():
    """Entry point for the application"""
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level
    )

if __name__ == "__main__":
    start_server()
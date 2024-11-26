"""
News Classification API Application Package
Contains core functionality for news classification and sentiment analysis.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
__version__ = "1.0.0"

# app/config.py
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "News Classification API"
    APP_VERSION: str = "1.0.0"
    MODEL_PATH: str = str(Path(__file__).parent.parent / "models")
    
    class Config:
        env_file = ".env"

settings = Settings()
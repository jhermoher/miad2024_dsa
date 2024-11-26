# config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "News Classification API"
    APP_VERSION: str = "1.0.0"
    MODEL_PATH: str = "./models"

    class Config:
        env_file = ".env"


settings = Settings()
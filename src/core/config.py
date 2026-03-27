import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Logo Masker"
    API_V1_STR: str = "/api/v1"
    
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    MODEL_PATH: str = os.getenv("MODEL_PATH", "runs/segment/logo_masker_model/weights/best.pt")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "yolo11n-seg.pt")
    
    CONF_THRESHOLD: float = 0.75
    IOU_THRESHOLD: float = 0.45
    
    UPLOAD_DIR: str = "data/uploads"
    RESULT_DIR: str = "data/results"

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)

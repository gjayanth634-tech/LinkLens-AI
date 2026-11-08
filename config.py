"""
Configuration settings for LinkLens AI application.
Loads configuration values from environment variables with sensible defaults.
"""

import os
from datetime import timedelta
import dotenv

# Load environment variables from .env file if available
dotenv.load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-linklens-ai-change-in-prod-12345")

    # Database Settings
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "linklens_ai")

    # Machine Learning Model Paths
    MODEL_PATH_RF = os.getenv("MODEL_PATH_RF", "models/random_forest.pkl")
    MODEL_PATH_XGB = os.getenv("MODEL_PATH_XGB", "models/xgboost_model.pkl")
    MODEL_PATH_CATBOOST = os.getenv("MODEL_PATH_CATBOOST", "models/catboost_model.pkl")

    # Session and Cookie Security
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "False").strip().lower() in ("true", "1", "t")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)

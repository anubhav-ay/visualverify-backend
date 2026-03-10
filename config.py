"""
Configuration module — loads all environment variables from .env file.
Falls back gracefully if API keys are missing (uses local models instead).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file from project root


class Settings:
    # ── Optional API Keys (system works without all of these) ────────────────
    BING_VISUAL_SEARCH_KEY: str = os.getenv("BING_VISUAL_SEARCH_KEY", "")
    GOOGLE_FACT_CHECK_KEY: str = os.getenv("GOOGLE_FACT_CHECK_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    MEDIASTACK_API_KEY: str = os.getenv("MEDIASTACK_API_KEY", "")

    # ── HuggingFace model names ───────────────────────────────────────────────
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    SENTENCE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── API base URLs ─────────────────────────────────────────────────────────
    GOOGLE_FACT_CHECK_URL: str = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    BING_VISUAL_SEARCH_URL: str = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
    NEWS_API_URL: str = "https://newsapi.org/v2/everything"
    MEDIASTACK_URL: str = "http://api.mediastack.com/v1/news"

    # ── Request timeout (seconds) ─────────────────────────────────────────────
    HTTP_TIMEOUT: int = 15

    # ── Data paths ────────────────────────────────────────────────────────────
    MISINFO_DATASET_PATH: str = "data/local_misinfo_dataset.json"


settings = Settings()

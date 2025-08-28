# src/core/settings.py
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)

NEWS_API_URL = os.getenv("NEWS_API_URL", "http://192.168.0.123:8000/news/extract")
EMB_MODEL    = os.getenv("EMB_MODEL", "BAAI/bge-base-en-v1.5")
HOST         = os.getenv("HOST", "0.0.0.0")
PORT         = int(os.getenv("PORT", "8001"))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "3500"))

ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://localhost:8001",
]

SERVICE_NAME = "redfin_target-insight"

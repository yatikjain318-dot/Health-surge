"""Application Configuration for SwasthyaSurge AI."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

APP_NAME = os.getenv("APP_NAME", "SwasthyaSurge AI")
APP_ENV = os.getenv("APP_ENV", "development")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "20260911"))
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", str(BASE_DIR / "outputs"))
PLOTS_DIR = os.getenv("PLOTS_DIR", str(BASE_DIR / "plots"))

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8050",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8050",
    "*"
]

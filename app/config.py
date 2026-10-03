import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "app" / "static"
ASSETS_DIR = STATIC_DIR / "assets"

# Ensure directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Load .env file
load_dotenv(BASE_DIR / ".env")

# App settings
APP_TITLE = "AutoInspect AI"
APP_DESCRIPTION = "Next-Gen Used Car Valuation & Visual Condition AI System"
APP_VERSION = "1.0.0"

# Gemini API settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Model paths
TABULAR_MODEL_PATH = WEIGHTS_DIR / "tabular_price_model.joblib"
VISION_MODEL_PATH = WEIGHTS_DIR / "cnn_damage_classifier.pth"
DATASET_PATH = DATA_DIR / "car_pricing_dataset.csv"

import os
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import (
    APP_TITLE, APP_DESCRIPTION, APP_VERSION,
    STATIC_DIR, ASSETS_DIR, GEMINI_API_KEY, GEMINI_MODEL
)
from app.models.tabular_model import tabular_predictor
from app.models.vision_model import vision_classifier
from app.models.llm_reporter import llm_reporter
from app.utils.sample_data import (
    POPULAR_MAKES, MODELS_BY_MAKE, TRANSMISSIONS, FUEL_TYPES,
    BODY_TYPES, YEARS, SAMPLE_CARS
)
from app.schemas.inspection import InspectionResponse, OptionsResponse

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

# Enable CORS for local development and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the main AutoInspect AI single-page application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)

@app.get("/api/health")
async def health_check():
    """Returns system status, active models, and Gemini configuration."""
    return {
        "status": "healthy",
        "app": APP_TITLE,
        "version": APP_VERSION,
        "tabular_model_loaded": tabular_predictor.pipeline is not None,
        "vision_model_loaded": vision_classifier.model is not None,
        "device": str(vision_classifier.device),
        "gemini_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY.strip()),
        "gemini_model": GEMINI_MODEL
    }

@app.get("/api/options", response_model=OptionsResponse)
async def get_filter_options():
    """Returns available vehicle makes, models, years, and powertrain options."""
    return {
        "makes": POPULAR_MAKES,
        "models_by_make": MODELS_BY_MAKE,
        "transmissions": TRANSMISSIONS,
        "fuel_types": FUEL_TYPES,
        "body_types": BODY_TYPES,
        "years": YEARS
    }

@app.get("/api/sample-cars")
async def get_sample_cars():
    """Returns the catalog of sample cars for instant 1-click evaluation."""
    return {"samples": SAMPLE_CARS}

@app.post("/api/inspect")
async def inspect_vehicle(
    make: str = Form("Toyota"),
    model: str = Form("RAV4"),
    year: int = Form(2021),
    mileage: float = Form(28500),
    engine_size: float = Form(2.5),
    transmission: str = Form("Automatic"),
    fuel_type: str = Form("Hybrid"),
    body_type: str = Form("SUV"),
    image: Optional[UploadFile] = File(None)
):
    """
    Main evaluation pipeline:
    1. Tabular ML: Predicts base resale valuation.
    2. PyTorch CNN: Classifies vehicle condition (Damaged vs. Whole).
    3. LLM Synthesis: Generates authoritative 3-paragraph inspection report.
    """
    specs = {
        "make": make.strip(),
        "model": model.strip(),
        "year": int(year),
        "mileage": float(mileage),
        "engine_size": float(engine_size),
        "transmission": transmission.strip(),
        "fuel_type": fuel_type.strip(),
        "body_type": body_type.strip()
    }

    # 1. Tabular Model Prediction
    try:
        base_price, range_low, range_high = tabular_predictor.predict_price(specs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tabular ML prediction error: {str(e)}")

    # 2. Vision CNN Condition Prediction
    try:
        if image and image.filename:
            image_bytes = await image.read()
            vision_result = vision_classifier.predict_image(image_bytes)
        else:
            # If no image provided, default to sample whole
            fallback_img = ASSETS_DIR / "sample_whole.jpg"
            vision_result = vision_classifier.predict_image(fallback_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision CNN prediction error: {str(e)}")

    # Calculate adjusted final valuation based on visual damage
    condition = vision_result["condition"]
    confidence = vision_result["condition_confidence"]
    condition_label = vision_result["condition_label"]
    damage_severity = vision_result["damage_severity"]
    penalty_pct = vision_result["damage_penalty_percent"]
    metrics = vision_result["metrics"]

    adjusted_price = round(base_price * (1.0 - penalty_pct), 2)
    adjusted_low = round(range_low * (1.0 - penalty_pct), 2)
    adjusted_high = round(range_high * (1.0 - penalty_pct), 2)

    # 3. LLM Synthesis (Gemini API 3.8 Flash)
    full_report, paragraphs, generated_by = llm_reporter.generate_report(
        specs=specs,
        tabular_price=base_price,
        condition=condition,
        confidence=confidence,
        damage_severity=damage_severity,
        adjusted_price=adjusted_price,
        metrics=metrics
    )

    inspection_id = f"INSP-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")

    return {
        "inspection_id": inspection_id,
        "timestamp": timestamp,
        "specs": specs,
        "base_predicted_price": base_price,
        "price_range_low": adjusted_low,
        "price_range_high": adjusted_high,
        "condition": condition,
        "condition_confidence": confidence,
        "condition_label": condition_label,
        "damage_severity": damage_severity,
        "damage_penalty_percent": round(penalty_pct * 100, 1),
        "adjusted_final_price": adjusted_price,
        "metrics": metrics,
        "inspection_report": full_report,
        "report_paragraphs": paragraphs,
        "generated_by": generated_by
    }

class SampleInspectRequest(BaseModel):
    sample_id: str

@app.post("/api/inspect/sample")
async def inspect_sample(request: SampleInspectRequest):
    """Inspects a built-in pre-loaded sample vehicle."""
    sample = next((s for s in SAMPLE_CARS if s["id"] == request.sample_id), None)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample car not found.")

    specs = {
        "make": sample["make"],
        "model": sample["model"],
        "year": sample["year"],
        "mileage": sample["mileage"],
        "engine_size": sample["engine_size"],
        "transmission": sample["transmission"],
        "fuel_type": sample["fuel_type"],
        "body_type": sample["body_type"]
    }

    base_price, range_low, range_high = tabular_predictor.predict_price(specs)
    
    img_path = ASSETS_DIR / sample["image_file"]
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Sample image file not found.")

    vision_result = vision_classifier.predict_image(img_path)

    condition = vision_result["condition"]
    confidence = vision_result["condition_confidence"]
    condition_label = vision_result["condition_label"]
    damage_severity = vision_result["damage_severity"]
    penalty_pct = vision_result["damage_penalty_percent"]
    metrics = vision_result["metrics"]

    adjusted_price = round(base_price * (1.0 - penalty_pct), 2)
    adjusted_low = round(range_low * (1.0 - penalty_pct), 2)
    adjusted_high = round(range_high * (1.0 - penalty_pct), 2)

    full_report, paragraphs, generated_by = llm_reporter.generate_report(
        specs=specs,
        tabular_price=base_price,
        condition=condition,
        confidence=confidence,
        damage_severity=damage_severity,
        adjusted_price=adjusted_price,
        metrics=metrics
    )

    inspection_id = f"INSP-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")

    return {
        "inspection_id": inspection_id,
        "timestamp": timestamp,
        "sample_id": sample["id"],
        "sample_title": sample["title"],
        "image_url": sample["image_url"],
        "specs": specs,
        "base_predicted_price": base_price,
        "price_range_low": adjusted_low,
        "price_range_high": adjusted_high,
        "condition": condition,
        "condition_confidence": confidence,
        "condition_label": condition_label,
        "damage_severity": damage_severity,
        "damage_penalty_percent": round(penalty_pct * 100, 1),
        "adjusted_final_price": adjusted_price,
        "metrics": metrics,
        "inspection_report": full_report,
        "report_paragraphs": paragraphs,
        "generated_by": generated_by
    }

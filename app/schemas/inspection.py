from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class CarSpecsInput(BaseModel):
    make: str = Field(..., example="Toyota")
    model: str = Field(..., example="RAV4")
    year: int = Field(..., ge=1990, le=2026, example=2021)
    mileage: float = Field(..., ge=0, example=32000)
    engine_size: float = Field(..., ge=0.5, le=8.0, example=2.5)
    transmission: str = Field("Automatic", example="Automatic")
    fuel_type: str = Field("Petrol", example="Petrol")
    body_type: str = Field("SUV", example="SUV")

class VisualInspectionMetrics(BaseModel):
    body_integrity_score: int = Field(..., ge=0, le=100)
    paint_condition_score: int = Field(..., ge=0, le=100)
    structural_score: int = Field(..., ge=0, le=100)
    market_desirability_score: int = Field(..., ge=0, le=100)
    composite_grade: str = Field(..., example="A (Excellent)")

class InspectionResponse(BaseModel):
    # Tabular predictions
    base_predicted_price: float
    price_range_low: float
    price_range_high: float
    
    # Vision CNN predictions
    condition: str  # "Whole" or "Damaged"
    condition_confidence: float
    condition_label: str
    damage_severity: str  # "None", "Minor", "Moderate", "Severe"
    damage_penalty_percent: float
    adjusted_final_price: float
    
    # Specs echo
    specs: Dict[str, Any]
    
    # Multi-point metrics
    metrics: VisualInspectionMetrics
    
    # LLM synthesized 3-paragraph report
    inspection_report: str
    report_paragraphs: List[str]
    generated_by: str
    
    # Timestamp / metadata
    inspection_id: str
    timestamp: str

class SampleCarItem(BaseModel):
    id: str
    title: str
    make: str
    model: str
    year: int
    mileage: float
    engine_size: float
    transmission: str
    fuel_type: str
    body_type: str
    expected_condition: str
    image_url: str
    description: str

class OptionsResponse(BaseModel):
    makes: List[str]
    models_by_make: Dict[str, List[str]]
    transmissions: List[str]
    fuel_types: List[str]
    body_types: List[str]
    years: List[int]

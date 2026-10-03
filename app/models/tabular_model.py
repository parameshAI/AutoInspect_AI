import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import TABULAR_MODEL_PATH, DATASET_PATH
from app.utils.sample_data import POPULAR_MAKES, MODELS_BY_MAKE, TRANSMISSIONS, FUEL_TYPES, BODY_TYPES

NUMERICAL_COLS = ["year", "mileage", "engine_size"]
CATEGORICAL_COLS = ["make", "model", "transmission", "fuel_type", "body_type"]

# MSRP Base pricing reference tier by make and model
BASE_MSRP = {
    "Toyota": {"RAV4": 32000, "Camry": 28000, "Corolla": 23000, "Highlander": 42000, "Tacoma": 35000, "Prius": 29000, "default": 30000},
    "Honda": {"Civic": 26000, "CR-V": 34000, "Accord": 30000, "Pilot": 43000, "HR-V": 27000, "default": 29000},
    "BMW": {"3 Series": 48000, "5 Series": 62000, "X3": 51000, "X5": 71000, "M340i": 59000, "default": 55000},
    "Mercedes-Benz": {"C-Class": 49000, "E-Class": 65000, "GLC": 52000, "GLE": 72000, "A-Class": 38000, "default": 58000},
    "Ford": {"F-150": 46000, "Mustang": 38000, "Explorer": 44000, "Escape": 31000, "Edge": 39000, "default": 38000},
    "Audi": {"A4": 46000, "A6": 61000, "Q5": 49000, "Q7": 68000, "A3": 37000, "default": 52000},
    "Hyundai": {"Elantra": 23000, "Sonata": 27000, "Tucson": 31000, "Santa Fe": 39000, "Kona": 26000, "default": 28000},
    "Tesla": {"Model 3": 43000, "Model Y": 49000, "Model S": 88000, "Model X": 96000, "default": 55000},
    "Volkswagen": {"Golf": 28000, "Jetta": 24000, "Tiguan": 33000, "Passat": 29000, "Atlas": 41000, "default": 31000},
    "Chevrolet": {"Silverado": 47000, "Malibu": 26000, "Equinox": 30000, "Tahoe": 60000, "Camaro": 37000, "default": 38000},
    "Nissan": {"Altima": 27000, "Rogue": 32000, "Sentra": 22000, "Pathfinder": 41000, "default": 28000}
}

def generate_synthetic_car_data(n_samples: int = 3500) -> pd.DataFrame:
    """Generates realistic used-car market data with real-world depreciation curves."""
    np.random.seed(42)
    records = []
    
    current_year = 2026
    
    for _ in range(n_samples):
        make = np.random.choice(POPULAR_MAKES)
        model = np.random.choice(MODELS_BY_MAKE.get(make, ["Sedan", "SUV"]))
        year = int(np.random.choice(list(range(2010, 2026)), p=[
            0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.09, 0.10, 0.10, 0.10, 0.08, 0.05, 0.01
        ]))
        age = current_year - year
        
        # Mileage correlated with age (~11k-14k miles per year + variance)
        mean_mileage = age * 12500
        mileage = float(max(500, np.random.normal(loc=mean_mileage, scale=mean_mileage * 0.28)))
        mileage = round(mileage, -2)
        
        # Engine size correlated with model/make
        if make == "Tesla":
            engine_size = 0.0  # Electric motor
            fuel_type = "Electric"
            transmission = "Automatic"
        else:
            engine_size = float(np.random.choice([1.5, 1.8, 2.0, 2.4, 2.5, 3.0, 3.5, 4.0, 5.0], 
                                                 p=[0.15, 0.15, 0.25, 0.15, 0.10, 0.10, 0.05, 0.03, 0.02]))
            fuel_type = np.random.choice(["Petrol", "Diesel", "Hybrid"], p=[0.70, 0.15, 0.15])
            transmission = np.random.choice(["Automatic", "Manual", "CVT"], p=[0.75, 0.15, 0.10])
            
        body_type = "Truck" if "F-150" in model or "Silverado" in model or "Tacoma" in model else (
            "SUV" if "RAV4" in model or "CR-V" in model or "X5" in model or "Q5" in model or "Explorer" in model else "Sedan"
        )
        
        # Calculate realistic base market price
        msrp_dict = BASE_MSRP.get(make, {})
        base_msrp = msrp_dict.get(model, msrp_dict.get("default", 32000))
        
        # Depreciation curve: 15% year 1, 10% next years, flattening at ~20% residual value
        depreciation_rate = 0.11
        age_factor = max(0.18, (1 - depreciation_rate) ** age)
        
        # Mileage impact (~$0.07 per mile difference from expected)
        expected_miles = age * 12000
        mile_diff = mileage - expected_miles
        mileage_adjustment = -0.065 * mile_diff
        
        # Engine boost for high displacement
        engine_bonus = (engine_size - 2.0) * 1200 if engine_size > 0 else 2500
        
        # Brand retention premium
        brand_multiplier = 1.0
        if make in ["Toyota", "Honda", "Porsche"]:
            brand_multiplier = 1.12
        elif make in ["BMW", "Mercedes-Benz", "Audi"]:
            brand_multiplier = 0.95 if age > 5 else 1.05
            
        price = max(2500.0, (base_msrp * age_factor + mileage_adjustment + engine_bonus) * brand_multiplier)
        # Add market noise
        noise = np.random.normal(0, max(50.0, price * 0.04))
        final_price = max(2000.0, price + noise)
        
        records.append({
            "make": make,
            "model": model,
            "year": year,
            "mileage": mileage,
            "engine_size": engine_size,
            "transmission": transmission,
            "fuel_type": fuel_type,
            "body_type": body_type,
            "price": round(final_price, 2)
        })
        
    df = pd.DataFrame(records)
    return df

class TabularPricePredictor:
    def __init__(self):
        self.pipeline: Pipeline = None
        self._load_or_train()

    def _build_pipeline(self) -> Pipeline:
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ("num", num_transformer, NUMERICAL_COLS),
            ("cat", cat_transformer, CATEGORICAL_COLS)
        ])
        
        # Random Forest Regressor
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        
        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

    def _load_or_train(self):
        if TABULAR_MODEL_PATH.exists():
            try:
                self.pipeline = joblib.load(TABULAR_MODEL_PATH)
                print(f"[TabularPredictor] Loaded trained model from {TABULAR_MODEL_PATH}")
                return
            except Exception as e:
                print(f"[TabularPredictor] Failed to load cached model: {e}. Retraining...")
        
        print("[TabularPredictor] Generating synthetic dataset and training Random Forest model...")
        df = generate_synthetic_car_data(n_samples=3200)
        df.to_csv(DATASET_PATH, index=False)
        print(f"[TabularPredictor] Saved dataset to {DATASET_PATH}")
        
        X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
        y = df["price"]
        
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        
        # Persist weights
        joblib.dump(self.pipeline, TABULAR_MODEL_PATH)
        print(f"[TabularPredictor] Successfully trained and saved model to {TABULAR_MODEL_PATH}")

    def predict_price(self, specs: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Accepts vehicle specifications dict and returns:
        (predicted_price, range_low, range_high)
        """
        # Prepare input dataframe with handling for missing values
        input_data = {
            "make": [specs.get("make", "Toyota")],
            "model": [specs.get("model", "RAV4")],
            "year": [int(specs.get("year", 2020))],
            "mileage": [float(specs.get("mileage", 45000))],
            "engine_size": [float(specs.get("engine_size", 2.0))],
            "transmission": [specs.get("transmission", "Automatic")],
            "fuel_type": [specs.get("fuel_type", "Petrol")],
            "body_type": [specs.get("body_type", "SUV")]
        }
        
        df_input = pd.DataFrame(input_data)
        predicted = float(self.pipeline.predict(df_input)[0])
        predicted = max(1500.0, round(predicted, 2))
        
        # Calculate fair market variance band (±7.5%)
        range_low = round(predicted * 0.925, 2)
        range_high = round(predicted * 1.075, 2)
        
        return predicted, range_low, range_high

# Global singleton
tabular_predictor = TabularPricePredictor()

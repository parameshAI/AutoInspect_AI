from typing import List, Dict, Any

POPULAR_MAKES = [
    "Toyota", "Honda", "BMW", "Mercedes-Benz", "Ford", 
    "Audi", "Hyundai", "Tesla", "Volkswagen", "Chevrolet", "Nissan"
]

MODELS_BY_MAKE = {
    "Toyota": ["RAV4", "Camry", "Corolla", "Highlander", "Tacoma", "Prius"],
    "Honda": ["Civic", "CR-V", "Accord", "Pilot", "HR-V"],
    "BMW": ["3 Series", "5 Series", "X3", "X5", "M340i"],
    "Mercedes-Benz": ["C-Class", "E-Class", "GLC", "GLE", "A-Class"],
    "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Edge"],
    "Audi": ["A4", "A6", "Q5", "Q7", "A3"],
    "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe", "Kona"],
    "Tesla": ["Model 3", "Model Y", "Model S", "Model X"],
    "Volkswagen": ["Golf", "Jetta", "Tiguan", "Passat", "Atlas"],
    "Chevrolet": ["Silverado", "Malibu", "Equinox", "Tahoe", "Camaro"],
    "Nissan": ["Altima", "Rogue", "Sentra", "Pathfinder"]
}

TRANSMISSIONS = ["Automatic", "Manual", "CVT", "Dual-Clutch"]
FUEL_TYPES = ["Petrol", "Diesel", "Hybrid", "Electric"]
BODY_TYPES = ["SUV", "Sedan", "Hatchback", "Truck", "Coupe"]
YEARS = list(range(2026, 2009, -1))

SAMPLE_CARS: List[Dict[str, Any]] = [
    {
        "id": "sample-whole-rav4",
        "title": "2021 Toyota RAV4 XLE (Pristine Condition)",
        "make": "Toyota",
        "model": "RAV4",
        "year": 2021,
        "mileage": 28500,
        "engine_size": 2.5,
        "transmission": "Automatic",
        "fuel_type": "Hybrid",
        "body_type": "SUV",
        "expected_condition": "Whole",
        "image_file": "sample_whole.jpg",
        "image_url": "/static/assets/sample_whole.jpg",
        "description": "Showroom quality SUV, garage kept, immaculate body panels and silver paint finish."
    },
    {
        "id": "sample-damaged-civic",
        "title": "2019 Honda Civic Sport (Front Collision Damage)",
        "make": "Honda",
        "model": "Civic",
        "year": 2019,
        "mileage": 44200,
        "engine_size": 1.5,
        "transmission": "Automatic",
        "fuel_type": "Petrol",
        "body_type": "Sedan",
        "expected_condition": "Damaged",
        "image_file": "sample_damaged.jpg",
        "image_url": "/static/assets/sample_damaged.jpg",
        "description": "Impact damage on front bumper, crumpled left fender, cracked grille, and hood deformation."
    }
]

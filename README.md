# AutoInspect AI 🚗🔍

AutoInspect AI is an end-to-end used-car evaluation system that combines **Tabular Machine Learning** (for price prediction), a **PyTorch Convolutional Neural Network** (for visual damage classification), and **Generative AI** (Gemini 3.8 Flash) to generate comprehensive vehicle inspection reports.

## Features

- **Tabular ML Pipeline**: Predicts vehicle value using a Scikit-learn Random Forest model trained on synthetic market data with realistic depreciation and mileage penalty curves.
- **Vision CNN**: A PyTorch Convolutional Neural Network with `AdaptiveAvgPool2d` that detects structural damage, computes a confidence score, and assigns a valuation penalty.
- **LLM Report Synthesis**: Generates a professional 3-paragraph inspection report using the Google Gemini API (or a robust heuristic fallback if offline).
- **Wow-Factor UI**: Modern glassmorphism dark-mode UI with drag-and-drop uploads, interactive charts, and text-to-speech audio reporting.
- **1-Click Demo Cars**: Includes pre-loaded samples (Whole vs Damaged) to instantly test the AI.

## Tech Stack

- **Backend**: FastAPI, Python 3.10+, Uvicorn
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Deep Learning**: PyTorch, Torchvision, Pillow
- **Generative AI**: `google-genai` SDK (Gemini 3.8 Flash)
- **Frontend**: HTML5, Vanilla JS, Custom CSS (Glassmorphism design)

## Local Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure API Key (Optional)**
Rename `.env.example` to `.env` and add your Gemini API key to unlock the LLM synthesis.
```env
GEMINI_API_KEY=your_api_key_here
```
*Note: If no key is provided, the app will seamlessly use its built-in rule-based AI synthesis engine!*

3. **Run the Application**
```bash
python run_local.py
```
The application will be available at `http://127.0.0.1:8000`. On first run, it will automatically generate the dataset, train the Random Forest model, initialize the PyTorch CNN, and save weights to the `/weights` directory.

## Deployment

### Containerized Deployment (Render, Railway, Fly.io, AWS)
A `Dockerfile` is included for zero-config containerized deployment.
```bash
docker build -t autoinspect-ai .
docker run -p 8000:8000 autoinspect-ai
```

### Vercel Deployment
A `vercel.json` is included. Note that Vercel has a 250MB serverless function limit. Since PyTorch is quite large, deploying on Vercel requires using PyTorch CPU-only wheels in your `requirements.txt` (`--index-url https://download.pytorch.org/whl/cpu`) or deploying the backend to a container service like Render/Railway and hosting the frontend on Vercel.

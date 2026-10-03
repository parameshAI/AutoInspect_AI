<div align="center">
  <img src="https://img.shields.io/badge/AutoInspect-AI-00F0FF?style=for-the-badge&logo=google-gemini&logoColor=white" alt="AutoInspect AI Badge"/>
  <h1>🚗 AutoInspect AI</h1>
  <p><strong>Next-Generation Multi-Modal Vehicle Valuation & Diagnostics</strong></p>

  <p>
    <a href="#features">Features</a> • 
    <a href="#architecture">Architecture</a> • 
    <a href="#tech-stack">Tech Stack</a> • 
    <a href="#quickstart">Quickstart</a>
  </p>
</div>

---

## 🌟 Overview

**AutoInspect AI** is an end-to-end, state-of-the-art used car evaluation system. It leverages a hybrid intelligence approach, combining traditional **Tabular Machine Learning** (for baseline market pricing) with cutting-edge **Computer Vision & Multi-Modal Generative AI** (for flawless structural damage detection). 

Simply upload a photo of a vehicle and input its specifications. The AI will instantly analyze the physical condition, detect any structural or exterior damage, adjust the market valuation, and generate an authoritative, executive-level inspection report.

## ✨ Features

- 👁️ **Hybrid Vision Pipeline:** Utilizes a custom PyTorch Convolutional Neural Network (CNN) feature extractor backed by **Gemini 3.8 Flash Multimodal API** acting as a flawless Oracle for absolute accuracy in damage classification.
- 📈 **Dynamic Valuation Model:** A Scikit-Learn Random Forest pipeline predicts baseline vehicle prices based on make, model, year, mileage, and powertrain, which is dynamically penalized based on visual damage severity.
- ⚡ **Cyber-Premium Interface:** A breathtaking, glassmorphic UI featuring 3D card tilts, custom animated laser scanners, and glowing radial gradients.
- 📄 **Executive AI Reporting:** Automatically synthesizes complex technical diagnostics into a readable 3-paragraph executive summary.

---

## 🧠 System Architecture

The application is built on a high-performance **FastAPI** backend that orchestrates three distinct AI models concurrently:

1. **Tabular Predictive Model (`scikit-learn`)**: Encodes vehicle specifications and outputs a baseline market price and estimated valuation range.
2. **Vision Classification Engine (`PyTorch` + `Gemini Multimodal`)**: Extracts image tensors and queries the Multimodal Oracle to definitively classify the vehicle as `WHOLE` or `DAMAGED`, assigning dynamic penalty scores to metrics like *Body Integrity* and *Structural Fit*.
3. **LLM Synthesis (`Google Gemini API`)**: Consumes the outputs of both models to generate a cohesive, human-readable inspection report.

---

## 🛠️ Tech Stack

### Artificial Intelligence & Machine Learning
- **Computer Vision:** PyTorch, Torchvision, Pillow
- **Tabular ML:** Scikit-Learn, Pandas, NumPy
- **Generative AI:** Google Gemini 3.8 Flash (Multimodal Vision & Text)

### Backend & API
- **Framework:** FastAPI
- **Server:** Uvicorn
- **Language:** Python 3.10+

### Frontend
- **Design:** Advanced Custom CSS (Glassmorphism, 3D CSS Transforms, CSS Animations)
- **Structure:** Vanilla HTML5 & JavaScript

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10 or higher
- A Google AI Studio API Key (`GEMINI_API_KEY`)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/parameshAI/AutoInspect_AI.git
   cd AutoInspect_AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your API key:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

4. **Run the server:**
   ```bash
   python run_local.py
   ```

5. **Open the App:** Navigate to `http://localhost:8000` in your web browser.

---

## 🐳 Docker Deployment (Render / Railway)

AutoInspect AI is containerized for instant deployment on cloud platforms.

1. Connect this repository to your preferred Docker hosting service (e.g., **Render**).
2. Add the `GEMINI_API_KEY` to the environment variables on your host.
3. Deploy! The included `Dockerfile` uses a lightweight Python 3.12-slim base image and automatically handles all necessary dependencies.

---

<div align="center">
  <p>Built with ❤️ and 🤖</p>
</div>

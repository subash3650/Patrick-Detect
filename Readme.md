# 🎤 Parkinson's Voice Detector

Advanced AI-powered voice analysis system for early detection of Parkinson's disease using machine learning and real-time audio processing.

---

## 📌 Overview

This full-stack application uses machine learning to analyze voice patterns and detect early signs of Parkinson's disease. The system records user voice samples, extracts advanced audio features using Praat/Parselmouth, and runs them through a trained Random Forest classifier.

**Key Features:**
- Real-time voice recording and analysis
- Advanced audio feature extraction (22 features)
- ML-based classification (Random Forest)
- Beautiful, responsive web interface
- REST API backend with Flask
- Production-ready code

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (Ubuntu/Linux/WSL)
- Node.js 14+ (for frontend)
- Git
- Microphone access (browser)

---

## 📦 Backend Setup (Python/Flask)

### 1. Install System Dependencies

sudo apt-get update
sudo apt-get install -y python3-dev python3-venv python3-pip
sudo apt-get install -y ffmpeg
sudo apt-get install -y libasound2-dev libsndfile1-dev
sudo apt-get install -y build-essential cmake

### 2. Create Virtual Environment

cd Backend
python3 -m venv venv
source venv/bin/activate

### 3. Install Python Dependencies

Upgrade pip
pip install --upgrade pip setuptools wheel
Install from requirements.txt
pip install -r requirements.txt

**Key packages:**
- Flask 3.0.0 - Web server
- praat-parselmouth 0.4.3 - Audio feature extraction
- scikit-learn 1.3.2 - ML model
- pydub 0.25.1 - Audio conversion
- joblib 1.3.2 - Model serialization

### 4. Verify Installation

python -c "
import flask
import flask_cors
import joblib
import pandas
import numpy
import scipy
import sklearn
import parselmouth
import pydub
import soundfile
print('✓ All packages installed successfully!')

### 5. Start Backend Server

cd Backend
source venv/bin/activate
python app.py

**Expected output:**
2025-11-15 11:31:40,643 - INFO - Starting Flask app...

Running on http://127.0.0.1:8000

Running on http://172.24.207.58:8000
Backend runs on: [**http://localhost:8000**](http://localhost:8000)

### Backend API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info & available endpoints |
| `/health` | GET | Health check |
| `/predict` | POST | Send audio for Parkinson's detection |

---

## 🎨 Frontend Setup (React/Node.js)

### 1. Install Dependencies

cd frontend
npm install
### 2. Start Development Server

npm start

Frontend runs on: [**http://localhost:3000**](http://localhost:3000)

The browser will automatically open the app.

### 3. Build for Production

npm build

Output will be in `frontend/build/` directory.

---

## 📁 Project Structure

PD/
│
├── Backend/
│ ├── venv/ # Virtual environment (not in git)
│ ├── app.py # Flask backend server
│ ├── extract_features_parselmouth.py # Audio feature extraction module
│ ├── data.py # Data processing utilities
│ ├── parkinsons_model.pkl # Trained ML model
│ ├── parkinsons.data # UCI Parkinson's dataset
│ ├── requirements.txt # Python dependencies
│ ├── .gitignore # Git ignore file
│ └── README.md
│
├── frontend/
│ ├── public/
│ │ └── index.html # HTML entry point
│ ├── src/
│ │ ├── App.js # Main React component
│ │ ├── App.css # Component styling
│ │ └── index.js # React DOM render
│ ├── package.json # npm dependencies
│ ├── package-lock.json # Dependency lock file
│ ├── .gitignore # Git ignore file
│ └── README.md
│
├── .gitignore # Root git ignore
└── README.md # This file


---

## 💻 Backend Technologies

### Framework & Server
- **Flask 3.0.0** - Lightweight web framework
- **Flask-CORS 4.0.0** - Cross-origin request handling

### Audio Processing
- **praat-parselmouth 0.4.3** - Praat speech analysis
- **pydub 0.25.1** - Audio format conversion
- **soundfile 0.12.1** - Audio file I/O
- **scipy 1.11.4** - Signal processing

### Machine Learning
- **scikit-learn 1.3.2** - ML algorithms & preprocessing
- **joblib 1.3.2** - Model persistence

### Data Processing
- **pandas 2.1.3** - Data manipulation
- **numpy 2.0.0** - Numerical computing

---

## 🎨 Frontend Technologies

### Framework & UI
- **React 19.2.0** - Component-based UI
- **React DOM 19.2.0** - DOM rendering

### HTTP & Styling
- **Axios 1.6.0** - HTTP client
- **CSS3** - Modern styling with animations

---

## 🔄 How It Works

### Audio Processing Pipeline

1.User Records Audio
↓
2.Browser Captures WebM Format
↓
3.Send to Backend via HTTP POST
↓
4.Convert WebM → WAV (pydub + ffmpeg)
↓
5.Extract Voice Features (Parselmouth)
├─ Pitch features (Fo, Fhi, Flo)
├─ Jitter measurements
├─ Shimmer measurements
├─ HNR (Harmonicity-to-Noise Ratio)
└─ Complex features (RPDE, DFA, etc.)
↓
6.Normalize Features (StandardScaler)
↓
7.ML Prediction (Random Forest)
↓
8.Return Results to Frontend
↓
9.Display Results with Confidence Score

### Machine Learning Model

**Model Type:** Random Forest Classifier
- **Trees:** 200
- **Features:** 6 critical voice parameters
- **Training Data:** UCI Parkinson's dataset
- **Test Accuracy:** ~96%

**Input Features:**
1. MDVP:Fo(Hz) - Average vocal fundamental frequency
2. MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
3. MDVP:Flo(Hz) - Minimum vocal fundamental frequency
4. MDVP:Jitter(%) - Variation in fundamental frequency
5. MDVP:Shimmer - Variation in amplitude
6. HNR - Harmonicity-to-Noise Ratio

---

## 📊 API Response Example

### Request
POST http://localhost:8000/predict
Content-Type: multipart/form-data
### Response (200 OK)
{
"prediction": 0,
"probability": 0.15,
"message": "Low likelihood of Parkinson's",
"meta": {
"duration": 5.2,
"samplerate": 44100
}
}
### Error Response (500)
{
"error": "Audio too short (0.3s). Please record at least 1 second.",
"trace": "..."
}

---

## 🎤 Using the Application

### Step 1: Start Backend
cd Backend
source venv/bin/activate
python app.py


### Step 2: Start Frontend (new terminal)
cd frontend
npm start


### Step 3: Use Web Interface
1. Open browser to `http://localhost:3000`
2. Click **"🎙 Start Recording (auto 5s)"**
3. Say a sustained **"aaaaaa"** sound
4. Wait for auto-stop or click **"⏹ Stop Recording"**
5. Click **"📤 Send for Analysis"**
6. View **Results** with confidence score

---

## 🛠️ Troubleshooting

### Backend Issues

#### Error: `ModuleNotFoundError: No module named 'parselmouth'`
Solution: Install with correct package name
pip install praat-parselmouth
Ubuntu/Debian
sudo apt-get install ffmpeg

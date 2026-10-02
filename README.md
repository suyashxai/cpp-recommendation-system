# 🌱 Crop Recommendation System

> **Machine Learning–Based Agricultural Crop Recommendation**  
> An ML Internship Project built with Python, Flask, and Scikit-learn

---

## Problem Statement

Farmers often make crop selection decisions based on limited information or experience, which can lead to poor yields. This project uses machine learning to analyse soil nutrient content and environmental conditions to recommend the most suitable crop — making precision agriculture accessible.

---

## Objectives

- Build a multi-class ML classification system for crop recommendation
- Analyse seven agricultural parameters (N, P, K, temperature, humidity, pH, rainfall)
- Compare four ML algorithms and select the best-performing model
- Deploy the trained model through a responsive Flask web application
- Provide a confidence score with every recommendation
- Store prediction history in a local SQLite database

---

## Features

| Feature | Description |
|---|---|
| 🤖 ML Recommendation | Trained Random Forest model on 2,200 agricultural records |
| 🎯 Confidence Score | Probability estimates for every prediction |
| 📊 Dashboard | Live statistics from real prediction data |
| 📋 History | Persistent prediction history in SQLite |
| 🌐 REST API | JSON API for programmatic access |
| ✅ Validation | Client-side and server-side input validation |
| 📱 Responsive UI | Works on desktop and mobile |

---

## ML Workflow

```
Dataset (CSV)
    ↓
Exploratory Data Analysis
    ↓
Preprocessing (drop duplicates, train/test split, StandardScaler)
    ↓
Model Training (Logistic Regression, Decision Tree, Random Forest, KNN)
    ↓
Model Evaluation (Accuracy, Precision, Recall, F1-score)
    ↓
Best Model Selected (based on weighted F1-score)
    ↓
Model Saved (crop_model.pkl + scaler.pkl)
    ↓
Flask Deployment
```

---

## Dataset

| Property | Value |
|---|---|
| Name | Crop Recommendation Dataset |
| Source | [Kaggle — Atharva Ingle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| Records | 2,200 |
| Features | N, P, K, temperature, humidity, ph, rainfall |
| Target | label (22 crop classes) |
| Format | CSV |

**Supported Crops:**  
apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, motherbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

> **Download the dataset** from Kaggle and place it at `data/Crop_recommendation.csv`

---

## Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Web Framework | Flask 3.x |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Model Storage | Joblib |
| Database | SQLite (built-in) |
| Frontend | HTML5, CSS3, JavaScript |
| Testing | pytest, pytest-flask |

---

## Project Structure

```
crop-recommendation-system/
│
├── app.py                        ← Flask entry point
├── requirements.txt              ← Python dependencies
├── README.md
│
├── data/
│   └── Crop_recommendation.csv   ← Dataset (place here)
│
├── model/
│   ├── train_model.py            ← Full ML training pipeline
│   ├── crop_model.pkl            ← Saved trained model (auto-generated)
│   └── scaler.pkl                ← Saved scaler (auto-generated)
│
├── backend/
│   ├── routes/
│   │   ├── main_routes.py        ← Home, About, History, Health
│   │   └── predict_routes.py     ← Predict (form + API)
│   ├── services/
│   │   ├── model_service.py      ← Model loading & prediction
│   │   └── db_service.py         ← SQLite operations
│   └── utils/
│       └── validation.py         ← Input validation
│
├── templates/
│   ├── index.html                ← Home + prediction form + dashboard
│   ├── result.html               ← Prediction result
│   ├── history.html              ← Prediction history table
│   ├── about.html                ← About page
│   ├── 404.html
│   └── 500.html
│
├── static/
│   ├── css/style.css
│   └── js/script.js
│
├── notebooks/
│   └── crop_analysis.ipynb       ← Jupyter notebook for exploration
│
├── tests/
│   ├── test_model.py             ← ML model tests
│   └── test_api.py               ← API & validation tests
│
├── reports/
│   ├── figures/                  ← Auto-generated visualisations
│   └── project_report.md         ← Full internship project report
│
└── screenshots/                  ← Add UI screenshots here
```

---

## Installation

### 1. Clone / download the project

```bash
git clone <repository-url>
cd crop-recommendation-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `Crop_recommendation.csv` from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) and place it at:

```
data/Crop_recommendation.csv
```

---

## How to Train the Model

```bash
python model/train_model.py
```

This will:
- Load and analyse the dataset
- Generate EDA visualisations in `reports/figures/`
- Train four ML models
- Print a comparison table
- Save the best model to `model/crop_model.pkl`
- Save the scaler to `model/scaler.pkl`

---

## How to Run the Application

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

---

## API Documentation

### `GET /api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "message": "Crop Recommendation API is running."
}
```

---

### `POST /api/predict`

Crop prediction endpoint.

**Request body:**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 25.5,
  "humidity": 80,
  "ph": 6.5,
  "rainfall": 200
}
```

**Success response (200):**
```json
{
  "success": true,
  "recommended_crop": "rice",
  "confidence": 0.94
}
```

**Error response (400):**
```json
{
  "success": false,
  "error": "Field 'ph' is out of valid range (0–14). Got: 15."
}
```

---

**Input Ranges:**

| Field | Min | Max | Unit |
|---|---|---|---|
| N | 0 | 140 | kg/ha |
| P | 5 | 145 | kg/ha |
| K | 5 | 205 | kg/ha |
| temperature | 0 | 50 | °C |
| humidity | 0 | 100 | % |
| ph | 0 | 14 | — |
| rainfall | 0 | 300 | mm |

---

## Testing

```bash
pytest tests/ -v
```

**Without a trained model** (validation tests only):
```bash
pytest tests/test_api.py::TestValidation -v
pytest tests/test_api.py::TestPageRoutes -v
```

**Full test suite** (requires trained model):
```bash
pytest tests/ -v
```

---

## Results

After running `python model/train_model.py`, a comparison table is printed to the console. Results are also saved in `reports/figures/evaluation_summary.txt`.

Typical results on the Crop Recommendation Dataset:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | ~96% | ~96% | ~96% | ~96% |
| Decision Tree | ~98% | ~98% | ~98% | ~98% |
| **Random Forest** | **~99%** | **~99%** | **~99%** | **~99%** |
| KNN | ~97% | ~97% | ~97% | ~97% |

> Actual numbers are printed during training and saved in `reports/figures/evaluation_summary.txt`.

---

## Limitations

- Recommendations are based on a static dataset; real-world conditions vary.
- The model does not account for regional crop varieties or soil type texture.
- Confidence scores reflect the model's certainty, not a guarantee of yield.
- The dataset covers 22 crop types; other crops are not supported.

---

## Future Improvements

- Real-time weather API integration (temperature, humidity, rainfall)
- GPS-based soil sensor integration
- Location-aware recommendations using regional datasets
- Explainable AI (SHAP/LIME feature explanations)
- Mobile application
- Multilingual interface
- Larger and more geographically diverse training datasets

---

## Disclaimer

This system provides **data-driven recommendations** based on patterns found in a publicly available agricultural dataset. It is **not a substitute for professional agricultural advice**. Always consult a qualified agronomist before making planting decisions.

---

## Author

**ML Internship Project**  
Built with Python, Flask, Scikit-learn, Pandas, NumPy, SQLite

---

## References

1. Crop Recommendation Dataset — Atharva Ingle, Kaggle:  
   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

2. Scikit-learn Documentation:  
   https://scikit-learn.org/stable/

3. Flask Documentation:  
   https://flask.palletsprojects.com/

4. Pandas Documentation:  
   https://pandas.pydata.org/docs/

5. NumPy Documentation:  
   https://numpy.org/doc/

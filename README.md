# 🌐 NetIntel — Hybrid Network Selection System

> Predict the best telecom provider for any location in India using real tower data, a CatBoost ML model, and a heuristic scoring engine — served via a Flask REST API with Gemini-powered chat.

---

## 📌 Overview

NetIntel is a hybrid network intelligence system that recommends the **best telecom provider** for a given location in India. It combines a **heuristic algorithm** (based on signal strength, tower proximity, and historical scores) with a **CatBoost ML model** trained on real-world tower and speed data to predict data speeds and rank providers.

The system exposes a **Flask REST API** with two endpoints:
- A direct lat/lon provider lookup
- A **Gemini-powered chatbot** (`Net Buddy`) that accepts natural language location queries

---

## 🗂️ Datasets Used

### 1. OpenCelliD
- Real cell tower data across India
- Features: Tower location (lat/lon), operator, tower range, sample count, signal metadata

### 2. India Telecom Internet Speed Dataset (2018–2023) — Kaggle
- Aggregated speed and signal data by operator and location (circle)
- Features: Average speed, signal stability, operator, geographic circle

---

## 🧠 ML Model — CatBoostRegressor

### Features Used
| Feature | Description |
|---|---|
| `Service Provider` | Telecom operator name |
| `Circle` | Indian telecom circle (state-level) |
| `provider_circle` | Combined operator + circle key |
| `signal_stability` | Stability score of signal |
| `signal_performance_ratio` | Derived ratio from signal strength |
| `prov_mean_speed` | Historical mean speed per operator |
| `prov_std_speed` | Historical std dev of speed per operator |
| `prov_med_signal` | Median signal metric per operator |

### Target
- **Predicted Data Speed** (Mbps)

### Why CatBoost?
- Handles categorical features (`operator`, `circle`) natively without encoding
- Robust on tabular telecom data with mixed types
- Fast inference suitable for real-time API calls

---

## ⚙️ Hybrid Scoring Engine

The provider ranking is **not purely ML** — it combines:

```
Combined Score = Predicted Speed (ML) + Heuristic Score (Tower Data)
```

**Heuristic Score** is derived from:
- Average distance to nearby towers (≤ 5 km radius, via Haversine formula)
- Estimated signal strength (log-scaled from sample count, distance, and tower range)
- Historical average score per operator from tower dataset

**Confidence Score** is computed for the top provider:
```
Confidence = 95 - (top_error / max_error) × 20
Clipped between 75% and 95%
```

---

## 🏗️ System Architecture

```
User Query (lat/lon or natural language)
        │
        ▼
  [Gemini NLP]  ──→  Extract Location
        │
        ▼
  Azure Maps API  ──→  Get Coordinates
        │
        ▼
  Tower Data (Azure Blob)
        │
        ├──→ Haversine Filter (≤5km towers)
        ├──→ Signal Strength Calculation
        └──→ Heuristic Score per Operator
                    │
                    ▼
          CatBoost Model (Azure Blob)
                    │
                    ▼
         Combined Score & Ranking
                    │
                    ▼
         Top 5 Providers + Confidence
                    │
                    ▼
       [Optional] Gemini Sentiment Summary
```

---

## 🚀 API Endpoints

### `POST /api/get_best_provider`
Returns the best provider for given coordinates.

**Request:**
```json
{
  "lat": 28.6139,
  "lon": 77.2090
}
```

**Response:**
```json
{
  "location": [28.6139, 77.2090],
  "provider": "Jio",
  "score": 91.4
}
```

---

### `POST /api/chatbot`
Natural language interface powered by Gemini.

**Request:**
```json
{
  "query": "Which network is best in Connaught Place Delhi?"
}
```

**Response:**
```json
{
  "location": "Connaught Place",
  "provider": "Jio",
  "score": 88.7,
  "sentiment": "Jio offers excellent coverage in Connaught Place with consistently high speeds..."
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | CatBoost Regressor |
| API Framework | Flask |
| Cloud Storage | Azure Blob Storage |
| Geocoding | Azure Maps API, Geopy (Nominatim) |
| NLP / Chat | Google Gemini 2.0 Flash |
| Data Processing | Pandas, NumPy |
| Distance Calc | Haversine Formula |

---

## 📦 Setup & Installation

```bash
git clone https://github.com/SidhaarthShree07/ML-SIM.git
cd netintel
pip install -r requirements.txt
```

Set environment variables:
```bash
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
export AZURE_MAPS_KEY="your_maps_key"
export GENAI_API_KEY="your_gemini_key"
```

Run the server:
```bash
python app.py
```

---

## 📁 Project Structure

```
netintel/
│
├── app.py                  # Main Flask application & API logic
├── requirements.txt        # Python dependencies
└── README.md
```

> **Note:** The CatBoost model (`speed_model_with_signal_v2.cbm`) and tower dataset (`cleaned_data.csv`) are stored and loaded dynamically from **Azure Blob Storage** for scalable cloud deployment.

---

## 👤 Author

**Sidhaarth Shree**
- 📧 sidhaarthshree@gmail.com

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

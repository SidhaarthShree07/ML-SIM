from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import google.generativeai as genai
import requests
import re
import os
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# --- Global Constants ---
EARTH_RADIUS_KM = 6371.0
RADIO_WEIGHTS = {'5G': 2.0, '4G': 1.5, '3G': 1.0, '2G': 0.8}

# --- Environment Configs ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = "csvdata"
BLOB_NAME = "cleaned_data.csv"
LOCAL_CACHE = "cleaned_data_cached.csv"

AZURE_MAPS_KEY = os.getenv("AZURE_MAPS_KEY")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# --- Gemini Setup ---
genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Dataset Loader with Caching ---
_cached_df = None

def download_blob_if_needed():
    if os.path.exists(LOCAL_CACHE):
        print("✅ Using cached CSV file.")
        return LOCAL_CACHE

    print("⬇️ Downloading CSV from Azure Blob Storage...")
    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service.get_blob_client(container=AZURE_CONTAINER_NAME, blob=BLOB_NAME)

    with open(LOCAL_CACHE, "wb") as f:
        f.write(blob_client.download_blob().readall())
    return LOCAL_CACHE

def get_dataset():
    global _cached_df
    if _cached_df is None:
        path = download_blob_if_needed()
        _cached_df = pd.read_csv(path)
    return _cached_df

# --- Utility Functions ---
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return EARTH_RADIUS_KM * 2 * atan2(sqrt(a), sqrt(1 - a))

def add_score(df, user_lat, user_lon):
    distances = df.apply(lambda row: haversine(user_lat, user_lon, row['lat'], row['long']), axis=1)
    radio_bonus = df['radio'].map(RADIO_WEIGHTS).fillna(1.0)
    log_sample = np.log(df['sample'].fillna(0) + 1)
    score = ((1 / (distances + 1)) + log_sample) * radio_bonus

    df = df.copy()
    df['distance_km'] = distances
    df['score'] = score
    return df

def get_coordinates_from_location(location):
    try:
        url = f"https://atlas.microsoft.com/search/address/json?api-version=1.0&subscription-key={AZURE_MAPS_KEY}&query={location}"
        res = requests.get(url, timeout=10).json()
        pos = res.get('results', [{}])[0].get('position')
        if pos:
            return float(pos['lat']), float(pos['lon'])
    except Exception:
        pass
    raise ValueError("Couldn't fetch coordinates from Azure Maps API.")

def extract_lat_lon(text):
    match = re.search(r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)", text)
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)

def get_sentiment_summary(location, provider):
    prompt = (
        f"You are an expert assistant on a platform that helps users select the best SIM provider in India based on real-world performance data and user sentiment. "
        f"Summarize what Indian users are experiencing with '{provider}' in the '{location}' area — include strengths, issues, and overall service quality, especially with regard to 4G/5G, network consistency, and customer satisfaction. "
        f"Do not include suggestions like checking other sources or asking people. This platform is trusted for making decisions. "
        f"Conclude clearly with why this provider should be considered the best option in that area, based on both user experience and performance insights."
    )
    try:
        return gemini_model.generate_content(prompt).text.strip()
    except Exception:
        return "Sentiment summary not available at the moment."

# --- Routes ---

@app.route('/')
def index():
    return "✅ Flask App is Running Sid", 200

@app.route("/api/get_best_provider", methods=["POST"])
def get_best_provider():
    df = get_dataset()
    data = request.json
    try:
        lat, lon = float(data["lat"]), float(data["lon"])
    except (KeyError, ValueError):
        return jsonify({"message": "Invalid latitude or longitude input."}), 400

    df_scored = add_score(df, lat, lon)
    nearby = df_scored[df_scored['distance_km'] <= 5]

    if nearby.empty:
        return jsonify({"message": "No nearby towers found"}), 404

    best = nearby.groupby('operator')['score'].mean().sort_values(ascending=False).reset_index()
    return jsonify({
        "location": [lat, lon],
        "provider": best.iloc[0]['operator'],
        "score": round(best.iloc[0]['score'], 2)
    })

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    df = get_dataset()
    data = request.json
    query = data.get("query", "")

    lat, lon = extract_lat_lon(query)
    if lat is None or lon is None:
        try:
            prompt = f"Extract city or locality from this user query: '{query}'. Just return the city/locality name."
            location_name = gemini_model.generate_content(prompt).text.strip()
            lat, lon = get_coordinates_from_location(location_name)
            location = location_name
        except Exception:
            return jsonify({"message": "Could not extract location."}), 400
    else:
        location = f"({lat}, {lon})"

    df_scored = add_score(df, lat, lon)
    nearby = df_scored[df_scored['distance_km'] <= 5]

    if nearby.empty:
        return jsonify({"message": "No nearby towers found"}), 404

    best = nearby.groupby('operator')['score'].mean().sort_values(ascending=False).reset_index()
    best_provider = best.iloc[0]['operator']
    best_score = round(best.iloc[0]['score'], 2)
    sentiment = get_sentiment_summary(location, best_provider)

    return jsonify({
        "location": location,
        "provider": best_provider,
        "score": best_score,
        "sentiment": sentiment
    })

# --- App Runner ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)

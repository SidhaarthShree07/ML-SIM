from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import google.generativeai as genai
import requests
import re
import os

app = Flask(__name__)
df = None

def get_dataset():
    global df
    if df is None:
        df = pd.read_csv("https://cleaneddata.blob.core.windows.net/csvdata/cleaned_data.csv")
    return df

# Azure and Gemini keys from environment (set them on Vercel)
AZURE_MAPS_KEY = os.environ.get("AZURE_MAPS_KEY")
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")

genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Distance using haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# Score towers
def add_score(df, user_lat, user_lon):
    df = df.copy()
    df['distance_km'] = df.apply(lambda row: haversine(user_lat, user_lon, row['lat'], row['long']), axis=1)

    radio_weights = {
        '5G': 2.0, '4G': 1.5, '3G': 1.0, '2G': 0.8
    }
    df['radio_bonus'] = df['radio'].map(radio_weights).fillna(1.0)

    df['score'] = (
        (1 / (df['distance_km'] + 1)) +
        np.log(df['sample'] + 1)
    ) * df['radio_bonus']
    return df

# Get lat/lon from city name using Azure Maps
def get_coordinates_from_location(location):
    url = f"https://atlas.microsoft.com/search/address/json?api-version=1.0&subscription-key={AZURE_MAPS_KEY}&query={location}"
    res = requests.get(url).json()
    if 'results' in res and len(res['results']) > 0:
        pos = res['results'][0]['position']
        return pos['lat'], pos['lon']
    else:
        raise ValueError("Couldn't fetch coordinates.")

# Get sentiment from Gemini
def get_sentiment_summary(location, provider):
    prompt = (
        f"You are an expert assistant on a platform that helps users select the best SIM provider in India based on real-world performance data and user sentiment. "
        f"Summarize what Indian users are experiencing with '{provider}' in the '{location}' area — include strengths, issues, and overall service quality, especially with regard to 4G/5G, network consistency, and customer satisfaction. "
        f"Do not include suggestions like checking other sources or asking people. This platform is trusted for making decisions. "
        f"Conclude clearly with why this provider should be considered the best option in that area, based on both user experience and performance insights."
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# Extract lat/lon from query
def extract_lat_lon(text):
    match = re.search(r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

@app.route('/')
def home():
    return "Welcome to SID ML PROJECT"
    
@app.route("/api/get_best_provider", methods=["POST"])
def get_best_provider():
    data = request.json
    lat = float(data["lat"])
    lon = float(data["lon"])

    df_scored = add_score(df, lat, lon)
    nearby = df_scored[df_scored['distance_km'] <= 5]

    if nearby.empty:
        return jsonify({"message": "No nearby towers found"}), 404

    best = nearby.groupby(['operator'])['score'].mean().sort_values(ascending=False).reset_index()
    best_provider = best.iloc[0]['operator']
    best_score = round(best.iloc[0]['score'], 2)

    return jsonify({
        "location": [lat, lon],
        "provider": best_provider,
        "score": best_score
    })

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    query = data["query"]

    lat, lon = extract_lat_lon(query)

    if lat is None or lon is None:
        prompt = f"Extract city or locality from this user query: '{query}'. Just return the city/locality name."
        location = gemini_model.generate_content(prompt).text.strip()
        lat, lon = get_coordinates_from_location(location)
    else:
        location = f"({lat}, {lon})"

    df_scored = add_score(df, lat, lon)
    nearby = df_scored[df_scored['distance_km'] <= 5]

    if nearby.empty:
        return jsonify({"message": "No nearby towers found"}), 404

    best = nearby.groupby(['operator'])['score'].mean().sort_values(ascending=False).reset_index()
    best_provider = best.iloc[0]['operator']
    best_score = round(best.iloc[0]['score'], 2)
    sentiment = get_sentiment_summary(location, best_provider)

    return jsonify({
        "location": location,
        "provider": best_provider,
        "score": best_score,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)

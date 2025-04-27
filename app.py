# -------------------------------
# Load Everything Once (Lazy Initialization)
# -------------------------------

import os
import re
import requests
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from flask import Flask, request, jsonify
from catboost import CatBoostRegressor
from geopy.geocoders import Nominatim
import google.generativeai as genai

app = Flask(__name__)

# -------------------------------
# Global Cache
# -------------------------------
model = None
tower_data = None
provider_stats = None
initialized = False 
# -------------------------------
# Configurations
# -------------------------------
BLOB_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "csvdata"
MODEL_BLOB_NAME = 'speed_model_with_signal_v2.cbm'
TOWERS_BLOB_NAME = "cleaned_data.csv"
AZURE_MAPS_KEY     = os.getenv("AZURE_MAPS_KEY")
GENAI_API_KEY      = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)
gemini_model       = genai.GenerativeModel('gemini-2.0-flash')
RESIDUAL_STD = 4.5
Z            = 1.96
EARTH_RADIUS_KM = 6371.0


# -------------------------------
# Utils
# -------------------------------
def load_dataframe(blob_name):
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_name)
    stream = io.BytesIO()
    stream.write(blob_client.download_blob().readall())
    stream.seek(0)
    return pd.read_csv(stream)

def load_model(blob_name):
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_name)
    stream = io.BytesIO()
    stream.write(blob_client.download_blob().readall())
    stream.seek(0)
    m = CatBoostRegressor()
    m.load_model(stream)
    return m

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return EARTH_RADIUS_KM * 2 * atan2(sqrt(a), sqrt(1 - a))

def calculate_signal_strength(d_km, samples, tower_range_m):
    base = -70
    loss = d_km * 5
    bonus = np.log1p(samples)*0.5
    penalty = (tower_range_m/1000)*2
    sig = base - loss + bonus - penalty
    return np.clip(sig, -110, -70)

def get_state_from_latlon(lat, lon):
    loc = Nominatim(user_agent="circle_locator").reverse((lat, lon), exactly_one=True)
    return loc.raw['address'].get('state') if loc and 'state' in loc.raw['address'] else None

def get_coordinates_from_location(location):
    url = f"https://atlas.microsoft.com/search/address/json?api-version=1.0&subscription-key={AZURE_MAPS_KEY}&query={location}"
    res = requests.get(url, timeout=10).json()
    pos = res.get('results', [{}])[0].get('position')
    if pos: return float(pos['lat']), float(pos['lon'])
    raise ValueError("Couldn’t fetch coordinates")

def extract_lat_lon(text):
    m = re.search(r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)", text)
    return (float(m[1]), float(m[2])) if m else (None, None)

def get_sentiment_summary(location, provider):
    prompt = (
        "Hello! you are Net Buddy, NetIntel Chat Assistant. "
        f"In 2–3 short sentences, tell the user why '{provider}' is a great choice in '{location}'."
    )
    try:
        return gemini_model.generate_content(prompt).text.strip()
    except:
        return "Sentiment summary not available at the moment."

# -------------------------------
# Lazy Initialization
# -------------------------------
def initialize():
    global model, tower_data, provider_stats, initialized
    if initialized: return

    model       = load_model(MODEL_BLOB_NAME)
    tower_data  = load_dataframe(TOWERS_BLOB_NAME)
    # provider_stats: historical mean/std of score for each operator
    ps = tower_data.groupby('operator')['score'].agg(['mean','std']).reset_index()
    ps.columns = ['Service Provider','prov_mean_speed','prov_std_speed']
    provider_stats = ps
    initialized = True

# -------------------------------
# Core provider-selection logic
# -------------------------------
def find_best_providers(lat, lon):
    initialize()
    circle = get_state_from_latlon(lat, lon)
    if not circle:
        return None, None, {'message':'Unable to detect Circle from location'}, 400

    # 1) distance + signal + heuristic
    df = tower_data.copy()
    df['distance_km'] = df.apply(
        lambda r: haversine(lat, lon, r['lat'], r['long']), axis=1
    )
    nearby = (
      df.loc[df['distance_km'] <= 5]
        .assign(
          signal_strength=lambda d: d.apply(
            lambda r: calculate_signal_strength(r['distance_km'], r['sample'], r['range']), axis=1
          )
        )
        .copy()
    )
    if nearby.empty:
        return None, None, {'message':'No nearby towers found'}, 404

    op_stats = nearby.groupby('operator').agg({
      'distance_km':'mean',
      'signal_strength':'mean',
      'score':'mean'
    }).reset_index().rename(columns={'score':'heuristic_score'})

    # 2) predict & build candidates
    cands = []
    for _, row in op_stats.iterrows():
        prov = row['operator']
        hs   = row['heuristic_score']
        spd  = row['signal_strength']
        dist = row['distance_km']

        ps = provider_stats[provider_stats['Service Provider']==prov]
        if ps.empty: 
            continue
        mean_s = ps['prov_mean_speed'].iloc[0]
        std_s  = ps['prov_std_speed'].iloc[0]

        inp = pd.DataFrame({
          'Service Provider':   [prov],
          'Circle':             [circle],
          'provider_circle':    [f"{prov}_{circle}"],
          'signal_stability':   [10.0],
          'signal_performance_ratio':[5.0/(spd+120)],
          'prov_mean_speed':    [mean_s],
          'prov_std_speed':     [std_s],
          'prov_med_signal':    [std_s]
        })
        pred_speed = model.predict(inp)[0]
        err        = abs(pred_speed - mean_s)
        combined   = pred_speed + hs

        cands.append({
          'provider':        prov,
          'combined_score':  combined,
          'pred_speed':      pred_speed,
          'speed_error':     err,
          'signal_strength': spd,
          'distance_km':     dist
        })

    if not cands:
        return None, None, {'message':'No valid providers'}, 404

    # 3) select top 5 by combined_score
    cands.sort(key=lambda x: x['combined_score'], reverse=True)
    top5 = cands[:5]

    # 4) compute confidence for the very top provider
    errs = [c['speed_error'] for c in cands]
    e_max = max(errs)
    top_err = top5[0]['speed_error']
    if e_max > 0:
        conf = 95 - (top_err / e_max) * (95 - 75)
    else:
        conf = 95.0
    conf = float(np.clip(conf, 75, 95))

    # 5) format the top5 list
    formatted = []
    for c in top5:
        formatted.append({
          'provider':         f"{c['provider']}_{circle}",
          'score':            round(c['combined_score'], 2),
          'signal_strength':  round(c['signal_strength'], 1),
          'distance_km':      round(c['distance_km'], 2)
        })

    return formatted, conf, None, None


@app.route('/api/get_best_provider', methods=['POST'])
def get_best_sim():
    data = request.get_json()
    lat, lon = data['lat'], data['lon']

    top5, confidence, err, code = find_best_providers(lat, lon)
    if err:
        return jsonify(err), code

    # take just the very top provider
    best = top5[0]

    return jsonify({
        "confidence":   round(confidence, 2),
        "lat":          lat,
        "lon":          lon,
        "distance_km":  best["distance_km"],
        "provider":     best["provider"]
    })


# -------------------------------
# /api/chatbot
# -------------------------------
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data  = request.get_json()
    query = data.get('query','')

    # 1) extract or geocode lat/lon
    lat, lon = extract_lat_lon(query)
    if lat is None:
        try:
            prompt     = f"Extract city or locality from this user query: '{query}'. Just return the city/locality name."
            loc_name   = gemini_model.generate_content(prompt).text.strip()
            lat, lon   = get_coordinates_from_location(loc_name)
            location   = loc_name
        except:
            return jsonify({"message":"Could not extract location."}), 400
    else:
        location = f"({lat},{lon})"

    # 2) call find_best_providers (which returns 4 values!)
    top5, confidence, err, code = find_best_providers(lat, lon)
    if err:
        return jsonify(err), code

    # 3) pick the very best
    best_provider = top5[0]['provider']

    # 4) compose sentiment
    sentiment = get_sentiment_summary(location, best_provider)

    # 5) return exactly the same JSON keys as your other endpoint
    return jsonify({
        "location":  location,
        "provider":  best_provider,
        "confidence": round(confidence, 2),
        "sentiment": sentiment
    })

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)

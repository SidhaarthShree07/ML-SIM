import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient
from catboost import CatBoostRegressor
from geopy.geocoders import Nominatim
import io
import os
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
RESIDUAL_STD = 4.5  # model's residual std (example)
Z = 1.96

# -------------------------------
# Utils
# -------------------------------
def load_blob_as_dataframe(blob_name):
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
    model = CatBoostRegressor()
    model.load_model(stream)
    return model

def calculate_signal_strength(distance_km, samples, tower_range_m):
    base_signal = -70
    signal_loss_distance = distance_km * 5
    sample_bonus = np.log1p(samples) * 0.5
    tower_range_penalty = (tower_range_m / 1000) * 2

    signal = base_signal - signal_loss_distance + sample_bonus - tower_range_penalty
    signal = np.clip(signal, -110, -70)
    return signal

def get_state_from_latlon(lat, lon):
    geolocator = Nominatim(user_agent="circle_locator")
    location = geolocator.reverse((lat, lon), exactly_one=True)
    if location and 'state' in location.raw['address']:
        return location.raw['address']['state']
    else:
        return None

# -------------------------------
# Load Everything Once (Lazy Initialization)
# -------------------------------
def initialize():
    global model, tower_data, provider_stats, blob_service_client, initialized

    if not initialized:
        print("Initializing app...")
        # Connect to Azure
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    
        # Load model and dataset
        model = load_model(MODEL_BLOB_NAME)
        tower_data = load_blob_as_dataframe(TOWERS_BLOB_NAME)
    
        # Precompute provider stats
        provider_stats_temp = tower_data.groupby('operator').agg({
            'score': ['mean', 'std'],
            'distance_km': 'median'
        })

        provider_stats_temp.columns = ['prov_mean_speed', 'prov_std_speed', 'prov_med_signal']
        provider_stats_temp = provider_stats_temp.reset_index().rename(columns={'operator': 'Service Provider'})
        provider_stats = provider_stats_temp.copy()
    
        initialized = True
        print("Initialization complete.")
    else:
        print("Initialization already done.")

@app.route('/')
def index():
    initialize()
    return "✅ Flask App is Running Sid", 200

# -------------------------------
# API Endpoint
# -------------------------------
@app.route('/api/get_best_provider', methods=['POST'])
def get_best_sim():
    data = request.get_json()
    latitude = data['lat']
    longitude = data['lon']

    # Initialize if not already done
    initialize()

    # Get Circle (state) from location
    circle = get_state_from_latlon(latitude, longitude)
    if not circle:
        return jsonify({'message': 'Unable to detect Circle from location'}), 400

    # Find nearby towers
    radius_km = 5
    towers_nearby = tower_data.copy()
    towers_nearby['distance_km'] = np.sqrt((towers_nearby['latitude'] - latitude) ** 2 +
                                           (towers_nearby['longitude'] - longitude) ** 2) * 111
    towers_nearby = towers_nearby[towers_nearby['distance_km'] <= radius_km]

    if towers_nearby.empty:
        return jsonify({'message': 'No nearby towers found'}), 404

    # Calculate signal strength
    towers_nearby['signal_strength'] = towers_nearby.apply(
        lambda row: calculate_signal_strength(row['distance_km'], row['samples'], row['range']),
        axis=1
    )

    # Group by operator
    operator_stats = towers_nearby.groupby('operator').agg({
        'distance_km': 'mean',
        'signal_strength': 'mean'
    }).reset_index()

    results = []
    for idx, row in operator_stats.iterrows():
        provider = row['operator']
        signal_strength = row['signal_strength']

        # Fetch provider stats
        prov_stat_row = provider_stats[provider_stats['Service Provider'] == provider]
        if prov_stat_row.empty:
            continue

        prov_mean_speed = prov_stat_row['prov_mean_speed'].values[0]
        prov_std_speed = prov_stat_row['prov_std_speed'].values[0]
        prov_med_signal = prov_stat_row['prov_med_signal'].values[0]

        # signal_stability: we don't have per sample here, so assume small value
        signal_stability = 2.0  # you can improve if you have signal stddev

        # Prepare model input
        model_input = pd.DataFrame({
            'Service Provider': [provider],
            'Circle': [circle],
            'provider_circle': [f"{provider}_{circle}"],
            'signal_stability': [signal_stability],
            'signal_performance_ratio': [5.0 / (signal_strength + 120)],
            'prov_mean_speed': [prov_mean_speed],
            'prov_std_speed': [prov_std_speed],
            'prov_med_signal': [prov_med_signal]
        })

        # Predict
        pred_speed = model.predict(model_input)[0]

        # Confidence
        ci_lower = pred_speed - Z * RESIDUAL_STD
        ci_upper = pred_speed + Z * RESIDUAL_STD
        ci_range = ci_upper - ci_lower
        confidence = max(0, min(100, 100 - (ci_range / pred_speed) * 100))

        results.append({
            'lat': latitude,
            'lon': longitude,
            'provider': provider,
            'confidence': round(confidence, 2),
            'distance_km': round(row['distance_km'], 2),
            'signal_strength': round(signal_strength, 1)
        })

    # Sort
    results = sorted(results, key=lambda x: x['confidence'], reverse=True)

    return jsonify(results)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)

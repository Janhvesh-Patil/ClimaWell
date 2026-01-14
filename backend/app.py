from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from datetime import datetime
import requests

app = Flask(__name__)

# CORS Configuration - Will add deployed frontend URL later
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:3000",
            "https://heat-aware-dashboard.vercel.app",
            "https://*.vercel.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# Load ML Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'heatwave_risk_model.pkl')

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

# Store latest sensor data in memory
latest_sensor_data = {
    "temperature": None,
    "timestamp": None,
    "status": "disconnected"
}

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'message': 'Heatwave Risk Prediction API',
        'sensor_status': latest_sensor_data['status'],
        'version': '1.0.0'
    })

@app.route('/sensor/temperature', methods=['POST', 'OPTIONS'])
def receive_sensor_data():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        temperature = data.get('temperature')

        if temperature is None:
            return jsonify({"error": "No temperature data"}), 400

        latest_sensor_data['temperature'] = float(temperature)
        latest_sensor_data['timestamp'] = datetime.now().isoformat()
        latest_sensor_data['status'] = 'connected'

        print(f"📡 Received sensor data: {temperature}°C at {latest_sensor_data['timestamp']}")

        return jsonify({
            "status": "success",
            "temperature": temperature,
            "received_at": latest_sensor_data['timestamp']
        }), 200

    except Exception as e:
        print(f"Error receiving sensor data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sensor/temperature', methods=['GET'])
def get_sensor_data():
    # Check if data is stale (older than 10 seconds)
    if latest_sensor_data['timestamp']:
        last_update = datetime.fromisoformat(latest_sensor_data['timestamp'])
        time_diff = (datetime.now() - last_update).total_seconds()

        if time_diff > 10:
            latest_sensor_data['status'] = 'disconnected'

    return jsonify(latest_sensor_data), 200

@app.route('/predict-with-sensor', methods=['POST', 'OPTIONS'])
def predict_with_sensor():
    if request.method == 'OPTIONS':
        return '', 204

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        use_sensor = data.get('use_sensor', False)  # Default to False since sensor not implemented

        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude required'}), 400

        # Fetch weather data from Open-Meteo API
        meteo_url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'daily': 'temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_gusts_10m_max,precipitation_sum',
            'timezone': 'auto',
            'forecast_days': 1
        }

        response = requests.get(meteo_url, params=params, timeout=10)
        meteo_data = response.json()

        daily = meteo_data['daily']

        # Use sensor temperature as max temp if available and requested
        if use_sensor and latest_sensor_data['temperature'] is not None:
            temp_max = latest_sensor_data['temperature']
            data_source = 'sensor'
        else:
            temp_max = daily['temperature_2m_max'][0]
            data_source = 'open-meteo'

        # Prepare features for model
        features = [
            temp_max,
            daily['temperature_2m_min'][0],
            daily['wind_speed_10m_max'][0],
            daily['wind_gusts_10m_max'][0],
            daily['precipitation_sum'][0]
        ]

        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        return jsonify({
            'risk': RISK_LABELS[prediction],
            'risk_code': int(prediction),
            'confidence': {
                'low': float(probabilities[0]),
                'medium': float(probabilities[1]),
                'high': float(probabilities[2])
            },
            'features_used': {
                'temperature_max': temp_max,
                'temperature_min': daily['temperature_2m_min'][0],
                'wind_speed_max': daily['wind_speed_10m_max'][0],
                'wind_gusts_max': daily['wind_gusts_10m_max'][0],
                'precipitation': daily['precipitation_sum'][0]
            },
            'data_source': data_source,
            'sensor_status': latest_sensor_data['status'],
            'sensor_note': 'Hardware functionality not yet implemented' if not use_sensor else None,
            'recommendations': get_recommendations(RISK_LABELS[prediction])
        })

    except Exception as e:
        print(f"Error in prediction with sensor: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        features = [
            data['temperature_2m_max'],
            data['temperature_2m_min'],
            data['wind_speed_10m_max'],
            data['wind_gusts_10m_max'],
            data['precipitation_sum']
        ]

        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        return jsonify({
            'risk': RISK_LABELS[prediction],
            'risk_code': int(prediction),
            'confidence': {
                'low': float(probabilities[0]),
                'medium': float(probabilities[1]),
                'high': float(probabilities[2])
            },
            'recommendations': get_recommendations(RISK_LABELS[prediction])
        })
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

def get_recommendations(risk_level):
    recommendations = {
        'Low': [
            "Conditions are safe for outdoor activities",
            "Stay hydrated with regular water intake",
            "Normal precautions are sufficient"
        ],
        'Medium': [
            "Stay hydrated - drink water every 30-45 minutes",
            "Limit outdoor activities during peak afternoon hours",
            "Wear light, loose-fitting clothing",
            "Take frequent breaks in shaded areas"
        ],
        'High': [
            "Stay indoors as much as possible",
            "Drink water every 20-30 minutes",
            "Avoid ALL outdoor activities between 11 AM - 5 PM",
            "Seek air-conditioned environments",
            "Know heatstroke symptoms: confusion, dizziness, nausea"
        ]
    }
    return recommendations.get(risk_level, [])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
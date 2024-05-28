from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import base64
from io import BytesIO
import requests
app = Flask(__name__)
CORS(app)

# Load the YAMNet model for general sound classification
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Function to load class names from CSV
def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_map_path = yamnet_model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

# Function to calculate RMS amplitude
def calculate_rms(audio_data):
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return rms

# Function to calculate Decibel
def calculate_db(rms_amplitude, ref=0.00002):
    db = 20 * np.log10(rms_amplitude / ref)
    return db

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ring')
def ring():
    print("Ring")

    # Send a request to the ESP32 to trigger the buzzer
    esp32_url = 'http://192.168.1.15:5000/activate_buzzer'
    try:
        response = requests.get(esp32_url)
        if response.status_code == 200:
            return 'Ring endpoint called successfully, ESP32 buzzer activated', 200
        else:
            return 'Failed to activate ESP32 buzzer', response.status_code
    except requests.exceptions.RequestException as e:
        return str(e), 500

@app.route('/receive_audio', methods=['POST'])
def receive_audio():
    audio_data = request.json.get('data')
    if not audio_data:
        return jsonify({"error": "No audio data provided"}), 400

    # Decode the Base64 encoded audio data
    audio_bytes = BytesIO(base64.b64decode(audio_data))

    # Convert the bytes to a NumPy array of the appropriate type
    audio_frames = np.frombuffer(audio_bytes.read(), dtype=np.int16)

    # Normalize the waveform to -1 to 1 range
    waveform = audio_frames / np.iinfo(np.int16).max

    # Save the waveform for later use
    global last_waveform
    last_waveform = waveform

    #print("Received and decoded audio data:")
    #print(f"Waveform shape: {waveform.shape}")
    #print(f"Waveform first 10 samples: {waveform[:10]}")

    return jsonify({"message": "Audio data received successfully"}), 200

@app.route('/classify_audio')
def classify_audio():
    sensitivity = float(request.args.get('sensitivity', 1))  # Default sensitivity is 1

    global last_waveform
    if last_waveform is None:
        return jsonify({"error": "No audio data available"}), 400

    waveform = last_waveform

    # Calculate RMS amplitude
    rms_amplitude = calculate_rms(waveform)
    # Calculate Decibel
    decibel = calculate_db(rms_amplitude)

    #print(f"RMS Amplitude: {rms_amplitude}")
    #print(f"Decibel: {decibel}")

    # Use YAMNet model to classify sound
    scores, _, _ = yamnet_model(waveform)
    mean_scores = scores.numpy().mean(axis=0)
    top_class_index = mean_scores.argmax()
    inferred_class = class_names[top_class_index]

    print(f"Inferred Class: {inferred_class}")
    #print(f"Scores: {mean_scores}")

    # Apply sensitivity adjustment to decibel level
    adjusted_decibel = decibel * sensitivity

    result = {
        "class": inferred_class,
        "rms": rms_amplitude,
        "db": adjusted_decibel
    }
    return jsonify(result)

if __name__ == '__main__':
    last_waveform = None
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, render_template, jsonify, request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import pyaudio
import math

app = Flask(__name__)

# Load the YAMNet model for general sound classification
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
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

# Function to calculate Decibell
def calculate_db(rms_amplitude, ref=0.00002):
    db = 20 * math.log10(rms_amplitude / ref)
    return db

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_audio')
def classify_audio():
    sample_rate = 16000
    record_duration = 1

    sensitivity = float(request.args.get('sensitivity', 1.0))

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)

    frames = []
    for _ in range(int(sample_rate * record_duration / 1024)):
        data = stream.read(1024)
        frames.append(data)

    audio_data = b''.join(frames)
    waveform = np.frombuffer(audio_data, dtype=np.int16) / np.iinfo(np.int16).max

    # Adjust waveform based on sensitivity
    waveform *= sensitivity

    # Calculate RMS amplitude
    rms_amplitude = calculate_rms(waveform)
    # Calculate Decibell
    decibel = calculate_db(rms_amplitude)
    # Use YAMNet model to classify sound
    scores, _, _ = yamnet_model(waveform)
    top_class_index = scores.numpy().mean(axis=0).argmax()
    inferred_class = class_names[top_class_index]

    stream.stop_stream()
    stream.close()
    p.terminate()

    result = {
        "class": inferred_class,
        "rms": rms_amplitude,
        "db": decibel,
        "waveform": waveform.tolist()  # Convert waveform to list
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

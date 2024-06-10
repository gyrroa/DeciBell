import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import librosa

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return audio, sr

def preprocess_audio(audio):
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio

def classify_audio(audio):
    scores, embeddings, spectrogram = yamnet_model(audio)
    scores_np = scores.numpy()
    return scores_np

def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names


mean_scores = np.mean(scores, axis=0)
top_class_index = np.argmax(mean_scores)
top_class_name = class_names[top_class_index]


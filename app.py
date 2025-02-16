from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import librosa
import os

app = Flask(__name__)

# Load the trained model (ensure path is correct)
try:
    model = load_model('models/model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        print("No file part")
        return "No file part", 400

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        print("No selected file")
        return "No selected file", 400

    # Save the uploaded file to the 'uploads' directory
    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)
    print(f"File saved to: {audio_path}")

    # Extract features from the uploaded audio file
    features = extract_features(audio_path)
    if features is None:
        print("Error extracting features")
        return "Error extracting features", 500

    features = np.expand_dims(features, axis=0)

    try:
        # Make a prediction using the loaded model
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)
        emotion = ['happy', 'anger', 'fear', 'sad'][predicted_class[0]]
        print(f"Prediction: {emotion}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction", 500

    return render_template('result.html', prediction=emotion)


def extract_features(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        print("Audio loaded successfully.")

        # Extract MFCCs with 40 coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        print(f"MFCCs extracted: {mfccs.shape}")

        # Ensure MFCCs have shape (40, time_steps), where time_steps = number of frames
        current_time_steps = mfccs.shape[1]

        # Pad or truncate MFCCs to ensure they match the expected number of time steps (174)
        expected_time_steps = 174

        if current_time_steps < expected_time_steps:
            # Pad the MFCC array along the time dimension
            padding = np.zeros((40, expected_time_steps - current_time_steps))  # Shape (40, padding_size)
            mfccs = np.concatenate((mfccs, padding), axis=1)  # Pad along the time axis
        elif current_time_steps > expected_time_steps:
            # Truncate the MFCC array along the time dimension
            mfccs = mfccs[:, :expected_time_steps]  # Only keep the first 174 time steps

        # Reshape the MFCCs to match the expected input shape (40, 174, 1)
        mfccs = mfccs.reshape(40, expected_time_steps, 1)  # Final shape should be (40, 174, 1)
        print(f"Final MFCC shape: {mfccs.shape}")

        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

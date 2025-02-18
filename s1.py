import os
import wave
import librosa.display
import pylab
import numpy as np

# Specify the directory paths for each class
class_paths = {
    'Positive': '### Path to Happy (Positive) samples',
    'Negative': '### Path to Sad (Negative) samples',
    'Fear': '### Path to Fear (Negative) samples (merged)',
    'Neutral': '### Path to Neutral samples'
}


# Dictionary to map class names to emotions
class_emotions = {
    'Class-1': 'Happy',
    'Class-2': 'Fear',
    'Class-3': 'Sad',
    'Class-4': 'Neutral'
}

# Function to preprocess audio file and extract features
def preprocess_audio(file):
    # Read audio file
    samples, sample_rate = librosa.load(file, sr=None)

    # Extract Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=1024, hop_length=512)
    mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

    return samples, mel_spectrogram_db, sample_rate

# Iterate over each class
for class_name, class_path in class_paths.items():
    wav_files = [os.path.join(class_path, file) for file in os.listdir(class_path) if file.endswith('.wav')]

    # Display only 2 graphs from each class
    for i in range(2):
        file = wav_files[i]
        samples, mel_spectrogram_db, sample_rate = preprocess_audio(file)

        # Display the waveform
        pylab.figure()
        pylab.plot(samples)
        pylab.title(f'{class_emotions[class_name]} - Audio waveform')

        # Display the Mel spectrogram
        pylab.figure()
        librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, hop_length=512, x_axis='time', y_axis='mel')
        pylab.title(f'{class_emotions[class_name]} - Mel Spectrogram')
        pylab.colorbar(format='%+2.0f dB')

        # Show both plots
        pylab.show()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
from sklearn.preprocessing import LabelEncoder

# Function to extract features from audio files
def extract_features(file_path, sample_rate=22050, n_fft=1024, hop_length=512, n_mfcc=13):
    # Load audio file
    audio, _ = librosa.load(file_path, sr=sample_rate)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    # Normalize features
    mfccs_normalized = np.mean(mfccs.T, axis=0)
    return mfccs_normalized

# Specify the directory path where your WAV files are located
dataset_folder = '##' #your dataset path

wav_files = []
labels = []  # Assuming binary classification, you need labels
for file in os.listdir(dataset_folder):
    if file.endswith('.wav'):
        wav_files.append(os.path.join(dataset_folder, file))
        # Extract label from file name
        label = file.split('_')[2]  # Assuming label is the third part separated by underscores
        labels.append(label)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Extract features and labels
X = []
y = labels_encoded  # Use the encoded labels
for file in wav_files:
    features = extract_features(file)
    X.append(features)

X = np.array(X)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the sequential model
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Use softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer-encoded labels
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
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
dataset_folder = '/Users/tej/PycharmProjects/NLP/AudioWAV'

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

# Build the sequential model with Adam optimizer
model_adam = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model with Adam optimizer
model_adam.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Build the sequential model with SGD optimizer
model_sgd = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model with SGD optimizer
model_sgd.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train both models
history_adam = model_adam.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
history_sgd = model_sgd.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

# Evaluate both models
test_loss_adam, test_acc_adam = model_adam.evaluate(X_test, y_test)
print('Adam Optimizer - Test accuracy:', test_acc_adam)

test_loss_sgd, test_acc_sgd = model_sgd.evaluate(X_test, y_test)
print('SGD Optimizer - Test accuracy:', test_acc_sgd)

# Plotting training and testing accuracy for both optimizers
plt.figure(figsize=(12, 6))

# Adam Optimizer
plt.plot(history_adam.history['accuracy'], label='Adam Optimizer - Training Accuracy', color='blue', linestyle='-')
plt.plot(history_adam.history['val_accuracy'], label='Adam Optimizer - Testing Accuracy', color='blue', linestyle='--')

# SGD Optimizer
plt.plot(history_sgd.history['accuracy'], label='SGD Optimizer - Training Accuracy', color='red', linestyle='-')
plt.plot(history_sgd.history['val_accuracy'], label='SGD Optimizer - Testing Accuracy', color='red', linestyle='--')

plt.title('Training and Testing Accuracy with Different Optimizers')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
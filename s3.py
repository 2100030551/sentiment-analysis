import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

def extract_features(file_path, max_pad_len=174):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Pad or truncate to max_pad_len
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs

def load_data(class_paths):
    labels = []
    features = []

    for class_label, folder_path in class_paths.items():
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.wav'):
                features.append(extract_features(file_path))
                labels.append(class_label)

    return np.array(features), np.array(labels)

class_paths = {
    'Positive': '### Path to Happy (Positive) samples',
    'Negative': '### Path to Sad (Negative) samples',
    'Fear': '### Path to Fear (Negative) samples (merged)',
    'Neutral': '### Path to Neutral samples'
}

# Load data
features, labels = load_data(class_paths)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Reshape data for compatibility with Conv2D layer
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Create a sequential model
model = Sequential()

# Add Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the data for dense layers
model.add(Flatten())

# Add Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_paths), activation='softmax'))

# Convert class labels to integer indices
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# Predict probabilities for the test set
probabilities = model.predict(X_test)

# Custom colors for each emotion class
colors = {
    'Positive': 'mediumseagreen',  # Positive sentiment (formerly Happy)
    'Negative': 'firebrick',       # Negative sentiment (formerly Sad or Fear)
    'Neutral': 'deepskyblue'       # Neutral sentiment (formerly Neutral)
}
...

# Plotting the predicted probabilities with custom colors
emotions = list(class_paths.keys())

fig, axs = plt.subplots(len(X_test), 1, figsize=(8, 4 * len(X_test)), sharex=True)

for i in range(len(X_test)):
    bars = axs[i].bar(emotions, probabilities[i], color=[colors[emotion] for emotion in emotions])
    axs[i].set_ylabel(f'Sample {i + 1}')

# Create a legend on the right side
fig.legend(emotions, title='Emotions', bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.xlabel('Emotions')

# Add a centered title below the x-axis
plt.suptitle("Sentiment Mapping: Positive, Negative, and Neutral Sentiments", x=0.5, y=0.92)

plt.show()

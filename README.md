# **Sentiment Analysis on Audio using NLP 🎤😃**

This Sentiment Analysis system uses Natural Language Processing (NLP) to predict the sentiment of audio content. The audio files are first transcribed into text and then analyzed using NLP techniques to predict sentiments such as **Positive**, **Negative**, or **Neutral**.

---

### Features 🌟

- **Audio Upload 🔊**: Users can upload audio files for sentiment analysis.
- **Audio Transcription 📝**: The uploaded audio is transcribed into text.
- **Sentiment Prediction 😄😞😐**: The system predicts sentiments such as **Positive**, **Negative**, and **Neutral** based on the transcribed text.
- **Web Interface 🌐**: Simple and user-friendly web interface for interaction.
- **Model Integration 🤖**: Pre-trained NLP model for sentiment classification on text data.

---

### Technologies Used ⚙️

- **Python 🐍**: The programming language used for the back-end.
- **Flask 🌐**: Web framework for building the app.
- **SpeechRecognition 🗣️**: For transcribing audio files to text.
- **TensorFlow 🤖**: For building and using NLP models for sentiment classification.
- **NLP Models (e.g., BERT, LSTM, etc.) 🧠**: For performing sentiment analysis on the transcribed text.
- **Numpy 🔢**: For numerical data handling.
- **Scikit-learn 📊**: For data preprocessing and model evaluation.
- **Matplotlib 📈**: For visualizing training results like accuracy and loss curves.

---

### Screenshots 📸

#### 1. **Audio File Upload**
![Sentiment Prediction](https://github.com/2100030551/sentiment-analysis/blob/main/screen%20shots/1.png)
  
*Description: This screenshot shows the file upload interface where users can upload their audio files for sentiment analysis.*

#### 2. **Sentiment Prediction**
![Sentiment Prediction](https://github.com/2100030551/sentiment-analysis/blob/main/screen%20shots/3.png)

*Description: After uploading the audio, the system displays the predicted sentiment (Positive, Negative, Neutral) based on the transcribed text.*

#### 3. **Model Training (Accuracy & Loss)**
![Model Training](https://github.com/2100030551/sentiment-analysis/blob/main/screen%20shots/2.png)
*Description: This graph shows the model's training process, including accuracy and loss values over the epochs.*

---

### How It Works ⚙️

1. **Preprocess Audio Data**: 
   - The audio file is first processed using the `SpeechRecognition` library to transcribe it into text.
   
2. **Sentiment Analysis on Text**: 
   - The transcribed text is fed into an NLP model (e.g., LSTM, BERT) trained to classify sentiments based on text features.
   
3. **Flask Web App**: 
   - The Flask app (`app.py`) serves as the interface for users to upload audio files.
   - After uploading, the app transcribes the audio, runs the sentiment analysis model on the transcribed text, and returns the sentiment prediction.

4. **Training Process**:
   - The NLP model is trained on labeled text data (Positive, Negative, Neutral) to predict the sentiment.
   - The training process is visualized using `Matplotlib`, showing how the model improves over epochs.

---

### Uploads Folder 📂

All audio files that are uploaded by users are saved in the `uploads/` folder.
This folder is used to temporarily store the audio files, which are then transcribed and processed for sentiment analysis.

---

### Model Training 🧠

To train the sentiment analysis model, you can use the `train_model.py` script. This script will:

- **Transcribe Audio**: Convert the audio to text using a speech recognition library.
- **Sentiment Classification**: Classify the transcribed text using an NLP model (e.g., LSTM or BERT).
- **Save the Trained Model**: Save the trained NLP model as `model.h5` for future use.

---

### Troubleshooting ⚠️

- **Audio File Not Uploaded**: Ensure that the audio file is in `.wav` or `.mp3` format and correctly uploaded.
- **Model Prediction Issues**: If the model isn't providing accurate predictions, check the model file (`model.h5`) and ensure it has been trained properly.
- **API Key Issues**: If you are using an external service or API to process the audio, ensure that your API keys (if applicable) are correctly configured in `app.py`.
- **Network Errors**: Ensure you have an active internet connection if required, as model predictions may involve server-side processing.

---

### Conclusion 🎉

This Sentiment Analysis on Audio project leverages NLP to classify sentiments such as Positive, Negative, and Neutral from audio content. The system first transcribes audio into text and then uses advanced NLP models for sentiment classification. With a simple Flask web interface, users can upload audio files and receive real-time predictions. This model can be easily extended for more languages and more detailed sentiment categories for improved accuracy.

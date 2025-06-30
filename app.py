import streamlit as st
import librosa
import numpy as np
import joblib

# Load the trained model
model = joblib.load('C:/Users/Lenovo/Desktop/Speech Emotion Recognition/emotion_model.pkl')

# App title
st.title("🎤 Speech Emotion Recognition App")
st.write("Upload a `.wav` file and I'll tell you the predicted emotion!")

# File uploader
uploaded_file = st.file_uploader("Choose a .wav audio file", type="wav")

# If a file is uploaded
if uploaded_file is not None:
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=22050)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(mfcc_mean)
    
    # Show result
    st.success(f"🧠 Predicted Emotion: **{prediction[0].capitalize()}**")

# Multilingual Few-Shot Speaker Identification System

A deep learning-based system for multilingual speaker identification and voice authentication using minimal enrollment data.

## 🧠 Project Overview

This project implements a few-shot speaker identification pipeline using ECAPA-TDNN architecture, designed to work efficiently with very limited voice samples. The system supports **English**, **Hindi**, and **Tamil** voice inputs and includes a web interface for real-time interaction.

### 🔍 Key Features
- Few-shot speaker recognition with high accuracy (93.3%)
- Supports multilingual voice input (Hindi, English, Tamil)
- Adaptive thresholding for speaker authentication
- Real-time web interface using Flask
- Cosine similarity-based matching for speaker verification

## 🗂️ Project Structure
'''
few-shot-multilingual-speaker-id/
├── model/ # Pretrained or intermediate models
├── static/ # Static assets for web interface (CSS, JS)
├── templates/ # HTML templates for Flask
├── app.py # Flask web application
├── ecapa_tdnn.py # ECAPA-TDNN model definition
├── preprocess.py # Audio preprocessing (MFCC, embedding, etc.)
├── README.md # Project documentation
├── requirements.txt # (Optional)
├── user_data.json # Stores enrolled user voice features

'''
## 🧾 Input & Output

- **Input**: Short audio samples of user speech (few seconds long).
- **Output**: Identified speaker's name and authentication status.

## 🌐 Supported Languages
- English
- Hindi
- Tamil

## 📊 Accuracy
- **93.3%** accuracy achieved on multilingual dataset with minimal enrollment per speaker.

## 📁 Files Description
- `app.py`: Main Flask server for the web interface
- `ecapa_tdnn.py`: ECAPA-TDNN model logic
- `preprocess.py`: Audio preprocessing pipeline (Librosa-based)
- `user_data.json`: JSON file storing voice embeddings for registered users


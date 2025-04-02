# Drowsiness Detection using Machine Learning

## Overview
This project implements a drowsiness detection system using machine learning and computer vision techniques. The system processes facial images to extract eye and mouth aspect ratio features and classifies the state of drowsiness using a Support Vector Machine (SVM) model.

## Features
- Detects drowsiness based on facial landmarks.
- Uses MediaPipe FaceMesh for facial landmark detection.
- Extracts eye and mouth aspect ratio features.
- Implements an SVM classifier for drowsiness detection.
- Supports model training and evaluation on a dataset.

## Technologies Used
- OpenCV
- dlib
- NumPy
- SciPy
- scikit-learn
- MediaPipe
- joblib

## Dataset
The project uses a dataset stored in Google Drive, structured as:
```
/content/drive/MyDrive/Dataset/
    ├── Drowsy/
    ├── Non-Drowsy/
```
Each folder contains images of drowsy and non-drowsy individuals, respectively.

## Installation
1. Install required dependencies:
    ```bash
    pip install opencv-python mediapipe scikit-learn joblib
    ```
2. Ensure you have access to the dataset stored in Google Drive.

## Usage
1. Mount Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
2. Load and preprocess the dataset.
3. Train the SVM model using extracted features.
4. Save and use the trained model for real-time detection.

## Model Training
- Features are extracted using the `extract_features()` function.
- The dataset is split into training and testing sets.
- Features are normalized using `StandardScaler()`.
- The SVM classifier is trained and saved using `joblib`.

## Future Improvements
- Implement deep learning-based classification for better accuracy.
- Add real-time video-based detection.
- Optimize feature extraction for faster performance.

## Author
Sreejan Dhar


# Drowsiness Detection using Machine Learning

## Overview
Drowsiness detection is a critical application in various fields, especially in transportation and workplace safety, where driver fatigue or worker exhaustion can lead to serious accidents. This project implements a drowsiness detection system using machine learning and computer vision techniques. The system processes facial images to extract essential features related to eye closure and mouth movements, which are then used to classify whether an individual is drowsy or alert.

## Features
- **Facial Landmark Detection**: Utilizes MediaPipe FaceMesh to detect and track facial landmarks in images.
- **Feature Extraction**: Computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to determine drowsiness.
- **Machine Learning Classification**: Implements a Support Vector Machine (SVM) classifier for classification.
- **Data Preprocessing**: Normalizes extracted features to enhance model performance.
- **Model Training and Evaluation**: Uses scikit-learn to train and evaluate the classifier.
- **Real-time Detection Potential**: Although this implementation processes static images, it can be extended for real-time video applications.

## Technologies Used
This project relies on the following technologies:
- **OpenCV**: For image processing and computer vision tasks.
- **dlib**: A powerful library for facial landmark detection.
- **NumPy & SciPy**: For numerical computations and distance calculations.
- **scikit-learn**: For machine learning model training and evaluation.
- **MediaPipe**: For efficient face landmark detection.
- **joblib**: For saving and loading trained machine learning models.

## Dataset
The project utilizes a dataset stored in Google Drive. The dataset is structured as follows:
```
/content/drive/MyDrive/Dataset/
    ├── Drowsy/      # Contains images of drowsy individuals
    ├── Non-Drowsy/  # Contains images of alert individuals
```
Each category contains multiple images labeled accordingly, which are used for training and evaluating the machine learning model.

## Installation
To set up the environment and run the project, follow these steps:
1. Install the required dependencies by running:
    ```bash
    pip install opencv-python mediapipe scikit-learn joblib
    ```
2. Ensure you have access to the dataset stored in Google Drive.
3. Mount Google Drive in your Google Colab environment:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Workflow
### 1. Data Preprocessing
- The dataset consists of images labeled as either drowsy or non-drowsy.
- Each image is read and processed to extract facial features using MediaPipe FaceMesh.
- The **eye aspect ratio (EAR)** and **mouth aspect ratio (MAR)** are computed from facial landmarks.

### 2. Feature Extraction
- **Eye Aspect Ratio (EAR)**: Measures eye openness using Euclidean distances between eye landmarks. EAR is computed using the formula:
    ```
    EAR = (||P2 - P6|| + ||P3 - P5||) / (2 * ||P1 - P4||)
    ```
  where P1-P6 represent key eye landmark points.
- **Mouth Aspect Ratio (MAR)**: Measures mouth openness, which increases during yawning, a common sign of drowsiness. MAR is computed as:
    ```
    MAR = (||P2 - P10|| + ||P4 - P8||) / (2 * ||P1 - P6||)
    ```
- These features are stored in a dataset for machine learning training.

### 3. Model Training
- The dataset is split into training and testing sets.
- Features are normalized using `StandardScaler()` to standardize the range of independent variables.
- An **SVM classifier** with a linear kernel is trained on the extracted features.
- The trained model is saved using `joblib` for future use.

### 4. Evaluation
- The trained SVM model is tested on unseen data.
- Performance metrics such as **accuracy, precision, recall, and F1-score** are computed to evaluate the classifier’s effectiveness.
- The confusion matrix is used to visualize classification performance.

## Future Improvements
- **Deep Learning Integration**: Implementing Convolutional Neural Networks (CNNs) or Transfer Learning for enhanced accuracy.
- **Real-time Detection**: Extending the project to work with live video streams for real-time monitoring.
- **Mobile Deployment**: Adapting the model for mobile applications to assist drivers or workers.
- **Multimodal Features**: Combining eye movement patterns, head posture, and other physiological signals for a more robust detection system.
- **Hyperparameter Tuning**: Fine-tuning SVM parameters and exploring ensemble learning techniques.

## Conclusion
This project provides a fundamental approach to drowsiness detection using computer vision and machine learning. The combination of **EAR and MAR metrics** with SVM classification demonstrates a computationally efficient method for fatigue detection. While currently applied to static image datasets, the system can be extended to real-time applications, contributing to **road safety, occupational health, and driver monitoring systems**.

## Acknowledgment
We would like to express our gratitude to:
- The developers of **MediaPipe** for providing an efficient facial landmark detection framework.
- The **OpenCV and scikit-learn** communities for their extensive libraries that made feature extraction and model training seamless.
- Publicly available datasets that facilitated the training and validation of this model.

## Author
Sreejan Dhar

For any questions or contributions, feel free to reach out!


<h1 align='center'>Brain Tumor Detector (Work In Progress) </h1>

## Overview
This project aims to assist in the diagnosis of brain tumors by building a CNN model trained on MRI images. The model predicts whether an MRI scan contains a tumor, and if so, what the classification of the tumor is. The architecture is designed to use custom CNN layers with techniques like data augmentation, dropout, and regularization to reduce overfitting.

## Installation
1. Clone this repository:
```
git clone https://github.com/brycemartin04/brain-tumor-detection
cd brain-tumor-detection
```
2. Install the required Python packages:
```
pip install -r requirements.txt
```
3. Create the necessary directories:
```
mkdir -p static/temp static/temp/predict
```
4. Run app.py and access the web app through the local host.
```
python app.py
```

## Project Status
This project is currently under development. I am working on improving the machine learning model to better classify different types of tumors. After this, I will turn my focus   to enhancing the web app to improve the user experience.

## Goals
- **Image Upload:** Let users upload existing images for detection / classification.
- **Model Prediction:** Use the deep learning model to predict the presence of a brain tumor.
- **Real-Time Feedback:** If a tumor is detected, return the classification of that tumor and the confidence score.

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript (for the user interface and camera integration).
- **Backend:** Python, Flask (for serving the model).
- **Model:** TensorFlow/Keras for the tumor detection and condition classification model.

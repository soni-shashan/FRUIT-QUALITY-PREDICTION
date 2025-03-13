# Fruit Quality Prediction 🍎🍌

Welcome to the **Fruit Quality Prediction** project! This application uses machine learning to analyze and predict the quality of various fruits using images. It's a great tool for anyone interested in using AI to assess fruit quality in real-time or through image uploads.

## Overview

This project is designed to predict the quality of fruits such as apples, bananas, guavas, limes, oranges, and pomegranates. The application utilizes a TensorFlow Lite model to classify fruits as "Good", "Bad", or "Mixed" based on uploaded images or real-time camera input.

## Features

- **Image Upload:** Upload an image of a fruit to get a quality prediction.
- **Single Snapshot:** Use a camera to take a photo and analyze it immediately.
- **Real-Time Detection:** Perform live fruit quality detection using your webcam.

## Installation

To get started with the Fruit Quality Prediction project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/soni-shashan/FRUIT-QUALITY-PREDICTION.git
   cd FRUIT-QUALITY-PREDICTION
    ```

2. **Install the necessary packages: Ensure you have Python 3.7 or later installed, then install the required packages using pip:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the model: Place the model.tflite file in the root directory of the project. This file contains the pre-trained TensorFlow Lite model used for predictions.**

### Usage
1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Choose an option:
- Upload Image: Select an image file to analyze.
- Single Snapshot: Take a photo using your device's camera.
- Real-Time Detection: Start the webcam to analyze fruit quality live.

3. View Results: The application will display the prediction result along with a confidence score and an emoji representing the quality.

### Customization
- **Model:** You can replace the model.tflite with a different TensorFlow Lite model if you have a custom-trained model for different fruit classes.
- **CSS Styling:** Modify the CSS in the app.py to change the look and feel of the Streamlit interface.

### Troubleshooting
- If the webcam is not accessible, ensure your device's camera permissions are enabled.
- For any errors related to the TensorFlow model, verify the model path and ensure it's correctly placed in the project directory.


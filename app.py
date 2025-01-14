from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import atexit
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = YOLO("runs/detect/V2-BW/weights/best.pt")

previous_image = None
previous_prediction = None

def cleanup_temp_folder():
    """Clean up all files in the static/temp folder on app close."""
    temp_folder = 'static/temp'
    predict_folder = 'static/temp/predict'
    if os.path.exists(temp_folder):
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        
        for filename in os.listdir(predict_folder):
            file_path = os.path.join(predict_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        print("Temp folder does not exist.")

# Register the cleanup function to be called on app close
atexit.register(cleanup_temp_folder)

@app.route('/', methods=['GET', 'POST'])
def index():
    labels = ''
    status = ''
    image_url = None
    confidence = None
    predicted_image_path= None
    global previous_image
    global previous_prediction

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result="No file part")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', result="No selected file")
        
        if file:
            # Save the uploaded image temporarily
            filename = secure_filename(file.filename)
            img_path = os.path.join('static/temp', filename)
            file.save(img_path)
            image_url = f'static/temp/{filename}'
           
            if previous_image != None:
                if os.path.exists(previous_image):
                    os.remove(previous_image)
                previous_image = None

            if previous_prediction != None:
                if os.path.exists(previous_prediction):
                    os.remove(previous_prediction)
                previous_prediction = None

            
            previous_image = f'static/temp/{filename}'

            if os.path.exists(img_path):
                results = model(image_url)
                if len(results[0].boxes.cls) !=0:
                    labels = results[0].boxes.cls[0].item()  # This will give you the class names
                    confidence = results[0].boxes.conf[0].item()  # Confidence scores for each prediction
                else: 
                    labels = "No Cancer Detected"

                if labels == 1:
                    labels = 'Meningioma'
                elif labels == 2:
                    labels = "Pituitary"
                elif labels == 0:
                    labels = "Glioma"

            

                # Get the predicted image from YOLO result (use OpenCV to work with the image)
                predicted_image = results[0].plot(labels=True, conf=False)  # Plot the predictions onto the image

                # Convert the result to a format we can save
                # 'predicted_image' is an array, so convert it to a PIL Image
                predicted_image_pil = Image.fromarray(predicted_image)

                # Save the predicted image manually
                predicted_image_path = os.path.join('static/temp/predict', filename)
                predicted_image_pil.save(predicted_image_path)

                # Set the URL for the predicted image
                predicted_image_path = f'static/temp/predict/{filename}'
                previous_prediction = predicted_image_path
            elif image_url != None:
                print (image_url)
                status = 'Error loading File, Please Try Again'

            
    
    return render_template('index.html', result=labels, status = status, predicted_image_path=predicted_image_path, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
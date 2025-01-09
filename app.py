from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import atexit
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = tf.keras.models.load_model('./models/binary_model.keras')

previous_image = None

def preprocess(path):
    image = Image.open(path)
    image = image.resize((224,224),Image.LANCZOS)
    image_values = np.array(image)
    image_values = image_values / 255.0
    # Ensure RGB format
    if len(image_values.shape) == 2:  # Grayscale image
        image_values = np.stack([image_values] * 3, axis=-1)
    elif image_values.shape[-1] == 1:  # Single-channel image
        image_values = np.repeat(image_values, 3, axis=-1)
    
    # Add batch dimension
    image_values = np.expand_dims(image_values, axis=0)
    return image_values

def cleanup_temp_folder():
    """Clean up all files in the static/temp folder on app close."""
    temp_folder = 'static/temp'
    if os.path.exists(temp_folder):
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path):
                    print(f"Deleting file: {file_path}")
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        print("Temp folder does not exist.")

# Register the cleanup function to be called on app close
atexit.register(cleanup_temp_folder)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    image_url = None
    confidence = None
    global previous_image
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
                os.remove(previous_image)
                previous_image = None

            # Prepare the image for prediction
            img_array = preprocess(img_path)

            previous_image = img_path

            # Make a prediction
            prediction = model.predict(img_array)
            confidence = prediction[0][0]

            # Classify based on the prediction
            result = 'Not Cancer' if prediction[0] > 0.5 else 'Cancer'

            if prediction[0] > 0.5:
                confidence = prediction[0][0]  # Use the raw probability for 'Cancer'
            else:
                confidence = 1 - prediction[0][0]
            
            confidence = float(confidence * 100)
            
    
    return render_template('index.html', result=result, image_url=image_url, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
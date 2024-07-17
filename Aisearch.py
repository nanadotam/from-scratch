import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
from flask import Flask, request, jsonify, render_template, redirect, flash, url_for
from PIL import Image as PILImage
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set upload directory
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB

# Create necessary directories
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists('frames'):
    os.makedirs('frames')

# Load the pre-trained model
model = InceptionV3(weights='imagenet')

@app.route('/')
def index():
    return render_template('index.html')

# Preprocess and predict method
def predict(img_path):
    # Process image path
    img = keras_image.load_img(img_path, target_size=(299, 299))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Predictions
    preds = model.predict(x)
    # Decode predictions == readable format
    return decode_predictions(preds, top=3)[0]

def extract_frames(video_path, output_folder):
    # Set video file path of input video with name and extension
    video = cv2.VideoCapture(video_path)

    # Extract images
    success, image = video.read() 
    count = 0
    while success:
        cv2.imwrite(f"{output_folder}/frame{count}.jpg", image)
        success, image = video.read()
        count += 1
    video.release()

# Specifies allowed file types
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if filename.endswith(('mp4', 'avi', 'mov')):
            # Extract frames from video
            extract_frames(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'frames')
        else:
            predictions = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify(predictions)

    return redirect(url_for('index'))

# Define the search_objects function
def search_objects(query):
    results = []
    frames_folder = 'frames'
    # Loop through all frames
    for frame_file in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_file)
        predictions = predict(frame_path)
        # Check if any of the predictions match the query
        for _, label, _ in predictions:
            if query.lower() in label.lower():
                results.append({
                    'frame': frame_file,
                    'predictions': predictions
                })
                break
    return results

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query:
        flash('No query provided')
        return redirect(request.url)
    
    results = search_objects(query)
    return render_template('search_results.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)

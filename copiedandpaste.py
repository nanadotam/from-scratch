import os
import numpy as np
import cv2
import base64
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\Users\Dell Inspiron\Documents\School_2024_2\Intro to AI\my_flask_app'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max file size: 100MB
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, (299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Make predictions with the model
def predict_class(img_array):
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Extract frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    
    if not cap.isOpened():
        # Handle case where video capture could not be opened
        print(f"Error opening video file: {video_path}")
        return frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((frame_idx, frame))
        frame_idx += 1
    
    cap.release()
    return frames

# Detect objects in frames and encode frames to base64
def detect_objects(frames):
    results = []
    for frame_idx, frame in frames:
        img_array = preprocess_image(frame)
        predictions = predict_class(img_array)
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        results.append({
            'frame_idx': frame_idx,
            'class': predictions[0][1],
            'probability': float(predictions[0][2]),
            'frame_base64': frame_base64
        })
    return results

# API endpoint to upload video and search for objects
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract frames from video
            frames = extract_frames(file_path)
            
            # Detect objects in frames and encode to base64
            results = detect_objects(frames)
            
            # Process search query
            search_query = request.form['search_query'].lower()
            filtered_results = []
            for result in results:
                if search_query in result['class'].lower():
                    filtered_results.append(result)
            
            if filtered_results:
                return render_template('results.html', results=filtered_results)
            else:
                flash('Object not found in video!')
                return redirect(request.url)
    
    return render_template('index.html')

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)

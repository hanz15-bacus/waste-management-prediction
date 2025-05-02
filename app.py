import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
MODEL_PATH = 'waste_classifier_model.h5'
model = load_model(MODEL_PATH)

# Get class names from directory structure
train_dir = 'DATASET/TRAIN'
class_names = sorted(os.listdir(train_dir))
print(f"Loaded classes: {class_names}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image and make prediction
        processed_image = preprocess_image(filepath)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        # Get top 3 predictions
        top_indices = prediction[0].argsort()[-3:][::-1]
        top_results = [
            {
                'class': class_names[i],
                'confidence': float(prediction[0][i])
            }
            for i in top_indices
        ]
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'top_results': top_results
        })
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
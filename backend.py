from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import base64

# Import your existing functions here
from Diagnosis import prepare_input, diagnose  # Assuming these functions are in Diagnosis.py

app = Flask(__name__)

# Configure file upload settings
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('images')

    predictions = []

    for file in files:
        if file and allowed_file(file.filename):
            # Convert image to PIL format
            img = Image.open(file.stream)

            # Preprocess the image and get prediction
            processed_img = prepare_input(img)  # Now send the image object to the prepare_input function
            prediction = diagnose(processed_img)  # Use your function to get prediction

            # Convert image to base64 to send it to the frontend
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            predictions.append({
                "image": img_str,
                "disease": prediction  # Assuming `prediction` is the disease class name
            })

    return render_template('index.html', predictions=predictions)

# Clear Route
@app.route('/clear', methods=['POST'])
def clear():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import base64


from Diagnosis import prepare_input, diagnose  

app = Flask(__name__)


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


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
            img = Image.open(file.stream)
            processed_img = prepare_input(img)  
            prediction = diagnose(processed_img)  
            # Convert image to base64 to send it to the frontend
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            predictions.append({
                "image": img_str,
                "disease": prediction
            })

    return render_template('index.html', predictions=predictions)

# Clear Route
@app.route('/clear', methods=['POST'])
def clear():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

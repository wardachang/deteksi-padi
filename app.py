from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils.predict import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model sekali saat server start
model = load_model('model/model_resnet_padi2.h5')

class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut",
    "Healthy"
]


@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')


@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/beranda_user', methods=['GET'])
def beranda_user():
    return render_template('beranda_user.html')

@app.route('/jadwal_user', methods=['GET'])
def jadwal_user():
    return render_template('jadwal_user.html')

@app.route('/riwayat_user', methods=['GET'])
def riwayat_user():
    return render_template('riwayat_user.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result, confidence = predict_image(
                filepath,
                model,
                class_names
            )

            return render_template(
                'index.html',
                prediction=result,
                confidence=confidence,
                img_path=filepath
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
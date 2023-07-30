from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import cv2

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from gevent.pywsgi import WSGIServer

# Define a Flask app
app = Flask(__name__, template_folder="template", static_url_path='/static')

# Model saved with Keras model.save()
MODEL_PATH = 'C:/Users/SATYADEVI/Desktop/MiniProject/project/Deployment/best_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')


def model_prediction(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    pred = np.argmax(model.predict(img))
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.getcwd()  # Set the base path to the current working directory
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)

        pred_class = model_prediction(file_path, model)
        CATEGORIES = ['Brownspot', 'Healthy']
        pred_label = CATEGORIES[pred_class]

        return pred_label

    return None


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

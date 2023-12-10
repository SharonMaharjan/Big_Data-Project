# app.py
# pip install -r requirements.txt

import os

from flask import Flask, request, make_response, jsonify
from werkzeug.utils import secure_filename
from fastai.vision.all import *
from fastai.data.external import *


# codeblock below is needed for Windows path #############
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
##########################################################

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

learner = load_learner('../resnet50best.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return {'error': 'no image found, in request.'}, 400

    file = request.files['image'] 
    if file.filename == '':
        return {'error': 'no image found. Empty'}, 400
 
    if file and allowed_file(file.filename): 
        filename = secure_filename(file.filename)
        img = PILImage.create(file)
        pred = learner.predict(img)
        print(pred)
        # if you want a json reply, together with class probabilities:
        # return jsonify(str(pred))
        # or if you just want the result
        return {'success': pred[0]}, 200

    return {'error': 'something went wrong.'}, 500

if __name__ == '__main__':
    port = os.getenv('PORT',5000)
    app.run(debug=True, host='0.0.0.0', port=port) 
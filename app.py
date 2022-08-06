import os
import numpy as np
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, session
from utils.functions import get_labels, get_prediction, load_model
from utils.extraction import crop_and_extract

config_path = 'model/yolov4-custom.cfg'
# weights_path = 'model/custom.weights'
labels_path = 'model/obj.names'

labels = get_labels(labels_path)    
model = load_model(config_path)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('first.html')    

@app.route('/predict', methods=['POST'])
def predict():
    if os.path.exists("static/detection.png"):
        os.remove("static/detection.png")

    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    img = Image.open(file_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    classes, scores, boxes = get_prediction(img, model, 0.4, 0.3)
    if(len(boxes) != 0):
        for box in boxes:
            (x, y, w, h) = box
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "{} : {:.2f}".format(labels[classes[0]], scores[0])
            
            (w1, h1), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            img = cv2.rectangle(img, (x, y - 25), (x + w1, y), (0, 0, 255), -1)
            img = cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)


        cv2.imwrite('static/detection.png', img)
        text = crop_and_extract(file_path, boxes)
    else:
        text = 'Number Plate not Detected'
    
    if os.path.exists(file_path):
        os.remove(file_path)

    return render_template('second.html', prediction_text = 'hi', image_path = 'static/detection.png')


if __name__ == '__main__':
    app.run()

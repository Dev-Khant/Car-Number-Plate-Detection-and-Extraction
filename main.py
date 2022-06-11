import cv2
import matplotlib
import numpy as np
from PIL import Image

from functions import get_labels, get_prediction, load_model
from extraction import crop_and_extract

def main():
    img_path = 'car2.jpg'
    config_path = 'model\yolov4-custom.cfg'
    weights_path = 'model\custom.weights'
    labels_path = 'model\obj.names'
    

    labels = get_labels(labels_path)
    model = load_model(config_path, weights_path)
    # color = get_colors(labels)

    img = Image.open(img_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    classes, scores, boxes = get_prediction(img, model, 0.4, 0.3)
    for box in boxes:
        (x, y, w, h) = box
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = "{} : {:.2f}".format(labels[classes[0]], scores[0])
        (w1, h1), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        img = cv2.rectangle(img, (x, y - 25), (x + w1, y), (0, 0, 255), -1)
        img = cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    cv2.imwrite('detection.png', img)

    print(crop_and_extract(img_path, boxes))

if __name__ == "__main__":
    main()


import cv2
from PIL import Image
import numpy as np
from utils.aws_textract import process_text_detection

def crop_and_extract(img, boxes):
    
    img = Image.open(img)
    for i, box in enumerate(boxes):
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        img = img.crop(box)  
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.resize(img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        gray = cv2.medianBlur(gray, 3)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
        dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

        cv2.imwrite('dilation.png', dilation)

    return process_text_detection('car-plate-extractor', 'dilation.png', 'ap-south-1')
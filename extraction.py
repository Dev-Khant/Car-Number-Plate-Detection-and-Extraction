import cv2
from PIL import Image
import numpy as np
import easyocr

def crop_and_extract(img, boxes):
    
    img = Image.open(img)
    for i, box in enumerate(boxes):
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        im = img.crop(box)  
        im.save(f'frame_{i}.png')

    reader = easyocr.Reader(["en"])
    result = reader.readtext('frame_0.png')

    return result if (len(result) != 0) else 'Number not clearly visible'
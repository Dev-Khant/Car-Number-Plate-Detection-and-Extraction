import os
import cv2
import numpy as np
import boto3

def get_labels(labels_path):
    # get the label class
    labels = open(labels_path).read().strip().split('\n')
    return labels

# def get_colors(labels):
#     # set the color for class label
#     color = np.full((len(labels), 3),[0, 0, 255], dtype='uint8')
#     return color

def load_model(config_path):
    
    print('========== Downloading Model ==========')
    if os.path.exists('new.weights'):
        pass
    else:
        session = boto3.Session(
                aws_access_key_id = 'AKIAQCDD7DGQCQ5KH5VP',
                aws_secret_access_key = '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'
            )

        s3 = session.resource('s3')
        s3.Bucket('car-plate-extractor').download_file('custom.weights', 'new.weights')
    print('========== Model Downloaded ==========')

    # load yolov4 model
    print('========== Loading Model ==========')
    net = cv2.dnn.readNet('new.weights', config_path)
    model = cv2.dnn_DetectionModel(net)

    model.setInputParams(size=(416, 416), scale = 1/ 255)
    print('========== Model Loaded ==========')
    return model

def get_prediction(img, model, confthres, nmsthres):

    return model.detect(img, nmsThreshold = nmsthres, confThreshold = confthres)

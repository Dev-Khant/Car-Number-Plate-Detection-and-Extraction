import cv2
import numpy as np

def get_labels(labels_path):
    # get the label class
    labels = open(labels_path).read().strip().split('\n')
    return labels

# def get_colors(labels):
#     # set the color for class label
#     color = np.full((len(labels), 3),[0, 0, 255], dtype='uint8')
#     return color

def load_model(config_path, weights_path):
    # load yolov4 model
    print('========== Loading Model ==========')
    net = cv2.dnn.readNet(weights_path, config_path)
    model = cv2.dnn_DetectionModel(net)

    model.setInputParams(size=(416, 416), scale = 1/ 255)
    print('========== Model Loaded ==========')
    return model

def get_prediction(img, model, confthres, nmsthres):

    return model.detect(img, nmsThreshold = nmsthres, confThreshold = confthres)
    # (H, W) = img.shape[:2]

    # # determine only the *output* layer names that we need from YOLO
    # # layers_names = net.getLayerNames()
    # ln = net.getUnconnectedOutLayersNames()

    # # construct a blob from the input image and then perform a forward
    # # pass of the YOLO object detector, giving us our bounding boxes and
    # # associated probabilities
    # blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
    #                              swapRB=True, crop=False)
    # net.setInput(blob)
    # layerOutputs = net.forward(ln)

    # # initialize our lists of detected bounding boxes, confidences, and
    # # class IDs, respectively
    # boxes = []
    # confidences = []
    # classIDs = []

    # # loop over each of the layer outputs
    # for output in layerOutputs:
    #     # loop over each of the detections
    #     for detection in output:
    #         # extract the class ID and confidence (i.e., probability) of
    #         # the current object detection
    #         scores = detection[5:]
    #         # print(scores)
    #         classID = np.argmax(scores)
    #         # print(classID)
    #         confidence = scores[classID]

    #         # filter out weak predictions by ensuring the detected
    #         # probability is greater than the minimum probability
    #         if confidence > confthres:
    #             # scale the bounding box coordinates back relative to the
    #             # size of the image, keeping in mind that YOLO actually
    #             # returns the center (x, y)-coordinates of the bounding
    #             # box followed by the boxes' width and height
    #             box = detection[0:4] * np.array([W, H, W, H])
    #             (centerX, centerY, width, height) = box.astype("int")

    #             # use the center (x, y)-coordinates to derive the top and
    #             # and left corner of the bounding box
    #             x = int(centerX - (width / 2))
    #             y = int(centerY - (height / 2))

    #             # update our list of bounding box coordinates, confidences,
    #             # and class IDs
    #             boxes.append([x, y, int(width), int(height)])
    #             confidences.append(float(confidence))
    #             classIDs.append(classID)
    
    # # apply non-maxima suppression to suppress weak, overlapping bounding
    # # boxes
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
    #                         nmsthres)

    # # ensure at least one detection exists
    # if len(idxs) > 0:
    #     # loop over the indexes we are keeping
    #     for i in idxs.flatten():
    #         # extract the bounding box coordinates
    #         (x, y) = (boxes[i][0], boxes[i][1])
    #         (w, h) = (boxes[i][2], boxes[i][3])

    #         # draw a bounding box rectangle and label on the image
    #         color = [int(c) for c in colors[classIDs[i]]]
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    #         text = "{} : {:.2f}".format(labels[classIDs[i]], confidences[i])
    #         (w1, h1), _ = cv2.getTextSize(
    #                 text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    #         img = cv2.rectangle(img, (x, y - 25), (x + w1, y), color, -1)
    #         img = cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    #         # print(boxes)
    #         # print(classIDs)




# Detect Car Number Plate & Extract the Number

This project is about detecting number plate of a car in an image and then extracting the number from the plate. <br>
**Tensorflow** and **OpenCV** are used for training model and for image processing.

## Model

Here model used for Object Detection is **YOLOv4** from DarkNet. And the library used for extracting text is **EasyOCR**. <br>
The accuracy achieved by YOLOv4 was **98.9%**.

## Deployment

**Flask** is used to develop Web App along with HTML and CSS. Have also used **Docker** to containerize flask app for deployment. <br>
Dockerized flask app is deployed on **AWS**. For deployment **EC2**, **ECR** and **ECS** are used.

#### Website is Deployed [HERE](http://ec2-13-233-123-32.ap-south-1.compute.amazonaws.com/). <br><br>

➤ In main page first we upload the image.

![Screenshot 2022-07-13 092843](https://user-images.githubusercontent.com/57898986/178670670-2bafd8f5-83b0-49ed-8ff0-d25bd5cb3c88.png) <br><br>

➤ On clicking **Detect** image will processed and the number will be extracted.

![Screenshot 2022-07-13 092916](https://user-images.githubusercontent.com/57898986/178671427-b81ad3e4-419d-45a1-a270-7242503bf7d2.png)




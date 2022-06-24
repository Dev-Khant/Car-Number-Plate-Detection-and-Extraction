import cv2
import easyocr
import os

img = cv2.imread('frame_0.png', 0)
gray = cv2.resize(img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
blur = cv2.GaussianBlur(gray, (5,5), 0)
gray = cv2.medianBlur(gray, 3)
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

cv2.imwrite('dilation.png', dilation)

reader = easyocr.Reader(["en"])
result = reader.readtext('dilation.png')
for i in result:
    if i[1] == 'IND':
        continue
    print(i[1].replace('',' '), end=' ')
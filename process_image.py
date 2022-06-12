import cv2
import easyocr
import os

img = cv2.imread('frame_0.png')
gray = cv2.resize(img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
blur = cv2.GaussianBlur(gray, (5,5), 0)
gray = cv2.medianBlur(gray, 3)
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

cv2.imwrite('dilation.png', dilation)

img = cv2.imread('dilation.png')

reader = easyocr.Reader(["en"])
result = reader.readtext(img)
for i in result:
    print(i[1], end=" ")
# try:
#     contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# except:
#     ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# for cnt in sorted_contours:
#     x,y,w,h = cv2.boundingRect(cnt)
#     height, width = img.shape

#     # if height of box is not a quarter of total height then skip
#     if height / float(h) > 6: continue
#     ratio = h / float(w)
#     # if height to width ratio is less than 1.5 skip
#     if ratio < 1.5: continue
#     area = h * w
#     # if width is not more than 25 pixels skip
#     if width / float(w) > 15: continue
#     # if area is less than 100 pixels skip
#     if area < 100: continue
#     # draw the rectangle

#     rect = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
#     roi = thresh[y-5:y+h+5, x-5:x+w+5]
#     roi = cv2.bitwise_not(roi)
#     roi = cv2.medianBlur(roi, 5)

    # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imwrite('temp.png', roi)
    # letter = reader.readtext('temp.png')
    # os.remove('temp.png')

    # cv2.imshow('frame', roi)
    # cv2.waitKey(0)
    # print(letter[0][1], end = " ")
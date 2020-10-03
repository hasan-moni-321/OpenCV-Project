# Loading necessary library
import cv2 as cv
import numpy as np


# Declaring size of webcam image
frameWidth = 640
frameHeight = 480
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# Declare color
myColor = [[5,107,0,19,255,255],
           [133,56,0,159,156,255],
           [57,76,0,100,255,255 ]]


# Function for finding colour
def findColor(img, myColor):
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    for color in myColor:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv.inRange(imgHSV, lower, upper)
        getContours(mask)
        #cv.imshow(str(color[0]), mask)


# Declaring Contour for color
def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>500:
            cv.drawContours(imgResult, cnt, -1, (255,0,0), 3)
            peri = cv.arcLength(cnt,True)
            approx= cv.approxPolyDP(cnt, 0.02*peri, True)
            x,y,w,h = cv.boundingRect(approx)


# Start webcam
while True:
    success, img = cap.read()
    imgResult = img.copy()
    findColor(img, myColor)

    cv.imshow('Result', imgResult)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

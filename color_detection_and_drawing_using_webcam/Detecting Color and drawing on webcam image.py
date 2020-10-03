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
           [57,76,0,100,255,255 ],
           [90,48,0,118,255,255],
           [57,34,60,200,220,255]]

# Declare color value
myColorValue = [[51,153,255], #BGR
                [255,0,255],
                [0,255,0],
                [255,0,0],
                [0,255,0]]


# Function for finding colour
def findColor(img, myColor, myColorValue):
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    count = 0
    newPints = []
    for color in myColor:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv.inRange(imgHSV, lower, upper)
        x,y = getContours(mask)
        cv.circle(imgResult, (x,y), 10, myColorValue[count], cv.FILLED)
        if x != 0 and y != 0:
            newPints.append([x,y,count])
        count += 1
        #cv.imshow(str(color[0]), mask)
    return newPints


# Declaring Contour for color
def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>500:
            #cv.drawContours(imgResult, cnt, -1, (255,0,0), 3)
            peri = cv.arcLength(cnt,True)
            approx= cv.approxPolyDP(cnt, 0.02*peri, True)
            x,y,w,h = cv.boundingRect(approx)
    return x+w//2, y


# Function for drawing on webcam image
def drawOnCanvas(myPoints, myColorValue):
    for point in myPoints:
        cv.circle(imgResult, (point[0], point[1]), 10, myColorValue[point[2]], cv.FILLED)


# Start webcam
myPoints = []   # [x,y, colorId]
while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPints = findColor(img, myColor, myColorValue)
    if len(newPints) != 0:
        for newP in newPints:
            myPoints.append(newP)
            
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValue)

    cv.imshow('Result', imgResult)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

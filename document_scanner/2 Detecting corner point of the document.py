# loading necessary library
import cv2 as cv
import numpy as np


frameWidth = 640
frameHeight = 480

cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)


# Image preprocessing
def imgPreprocessing(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv.Canny(imgBlur, 200, 200)

    kernel = np.ones((5,5))
    imgDial = cv.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv.erode(imgDial, kernel, iterations=1)
    return imgThres


def finding_contours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 5000:
            #cv.drawContours(imgContour, cnt, -1, (255, 0,0), 3)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    # selecting only biggest one
    cv.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

def only_document(img, biggest):
    pass


while True:
    success, img = cap.read()
    img = cv.resize(img, (frameWidth, frameHeight))
    imgContour = img.copy()
    imgThres = imgPreprocessing(img)
    biggest = finding_contours(imgThres)
    print(biggest)
    only_document(img, biggest)

    cv.imshow('Reult', imgContour)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break




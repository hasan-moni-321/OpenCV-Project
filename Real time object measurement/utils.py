# Loading necessary library
import cv2 as cv
import numpy as np

# Function for finding contours
def getContours(img, cThr=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5,5))
    imgDial = cv.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv.erode(imgDial, kernel, iterations=2)
    if showCanny: cv.imshow("Canny", imgThre)

    contours, hierarchy = cv.findContours(imgThre, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    finalCountours = []

    for i in contours:
        area = cv.contourArea(i)
        if area > minArea:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02*peri, True)
            bbox = cv.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox,i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x:x[1], reverse=True)
    
    if draw:
        for con in finalCountours:
            cv.drawContours(img, con[4], -1, (0,0,255), 3)
    return img, finalCountours

# Function for reorder of the points
def reOrder(myPoints):
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def warpImg(img, points, w,h, pad=20):
    #reOrder(points)
    points = reOrder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv.warpPerspective(img, matrix, (w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    return imgWarp

# Function for finding Distance of side
def findDis(pts1, pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5
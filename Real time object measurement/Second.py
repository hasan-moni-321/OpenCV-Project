# Loading necessary library
import numpy as np
import cv2 as cv
import utils

# Declaring some variable
webcam = False
path = "ROM.jpg"
cap = cv.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)
scale = 3
wP = 210*3
hP = 297*3


# Use webcam or not
while True:
    if webcam: ret, img = cap.read()
    else: img = cv.imread(path)
    # calling getContours function for finding A4 size image
    img, conts = utils.getContours(img, minArea=50000, filter=4)

    if len(conts) != 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = utils.warpImg(img, biggest, wP, hP)
        # calling getContours function for finding objects in A4 image
        img2, conts2 = utils.getContours(imgWarp,
                                         minArea=2000, filter=4,
                                         cThr=[50,50], draw=False)

        if len(conts2) != 0:
            for obj in conts2:
                cv.polylines(img2, [obj[2]], True, (0,255,0), 2)
                nPoints = utils.reOrder(obj[2])
                nW = round((utils.findDis(nPoints[0][0]//scale, nPoints[1][0]//scale)/10), 1)
                nH = round((utils.findDis(nPoints[0][0]//scale, nPoints[2][0]//scale)/10), 1)

                cv.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                               (255, 0, 255), 3, 8, 0, 0.05)
                cv.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                               (255, 0, 255), 3, 8, 0, 0.05)

                x,y,w,h = obj[3]
                cv.putText(img2, '{}cm'.format(nW), (x+30, y-10), cv.FONT_HERSHEY_COMPLEX_SMALL,1.5, (255,0,255),2)
                cv.putText(img2, '{}cm'.format(nH), (x-70, y+h//2), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,0,255), 2)


        cv.imshow('img2', img2)

    img = cv.resize(img, (0,0), None, .5,.5)
    cv.imshow('Original', img)
    if cv.waitKey(1) == ord('q'):
        break


# Release everything if job is finished
cap.release()
cv.destroyAllWindows()


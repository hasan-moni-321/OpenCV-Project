# Loading necessary library
import numpy as np
import cv2 as cv

# Declaring variable
frameWidth = 640
frameHeight = 480
nPlateCascade = cv.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 500
color = (255,0,255)

# Defining image capture
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0


while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    numberPlate = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x,y, w,h) in numberPlate:
        area = w*h
        if area> minArea:
            cv.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)
            cv.putText(img, 'number plate', (x,y-5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            numberPlateRegion = img[y:y+h, x:x+w]
            cv.imshow('only plate image', numberPlateRegion)

    cv.imshow('frame', img)
    if cv.waitKey(1) == ord('q'):
        # saving images in folder
        cv.imwrite("ImageStore"+str(count)+".jpg", numberPlateRegion)
        cv.rectangle(img, (0,150), (640,300), (0,255,0), cv.FILLED)
        cv.putText(img, 'saved', (150,265), cv.FONT_HERSHEY_DUPLEX, 2, (0,255,255),2)
        cv.imshow('Result', img)
        cv.waitKey(500)
        count +=1

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



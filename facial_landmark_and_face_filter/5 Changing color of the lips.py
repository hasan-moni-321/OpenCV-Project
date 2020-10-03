
# Adding trackbar


# Loading necessary library
import numpy as np
import cv2 as cv
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def empty(a):
    pass


cv.namedWindow("BGR")
cv.resizeWindow("BGR", 640, 240)
cv.createTrackbar("Blue", "BGR", 0, 255, empty)
cv.createTrackbar("Green", "BGR", 0, 255, empty)
cv.createTrackbar("Red", "BGR", 0, 255, empty)


def createBox(img, points, scale=5, masked=False, cropped=True):
    if masked:
        # Finding mask image
        mask = np.zeros_like(img)
        mask = cv.fillPoly(mask, [points], (255,255,255))
        # taking original leaf not mask leaf
        img = cv.bitwise_and(img, mask)
        #cv.imshow("Mask", img)

    if cropped:
        bbox = cv.boundingRect(points)
        x,y,w,h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv.resize(imgCrop, (0,0), None, scale, scale)
        return imgCrop
    else:
        return mask

while True:
    img = cv.imread("taylor swift.jpg")
    img = cv.resize(img, (480,380), None, 0.5,0.5)
    imgOriginal = img.copy()

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        #imgOriginal = cv.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        landmarks = predictor(imgGray, face)
        #landmark point
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
            #cv.circle(imgOriginal, (x,y), 5, (50,50,255), cv.FILLED)
            #cv.putText(imgOriginal, str(n), (x,y-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1)
        ##### seperating left eye
        #myPoints = np.array(myPoints)
        #imgLeftEye = createBox(img, myPoints[36:42], 3)
        #print(myPoints)
        ##### seperating leaf
        myPoints = np.array(myPoints)
        imgLips = createBox(img, myPoints[48:61], 3, masked=True, cropped=False)

        # creating a new image with new color
        imgColorLips = np.zeros_like(imgLips)
        b = cv.getTrackbarPos("Blue", "BGR")
        g = cv.getTrackbarPos("Green", "BGR")
        r = cv.getTrackbarPos("Red", "BGR")
        imgColorLips[:] = b, g, r
        # changing color of the lips with with newly created image color
        imgColorLips = cv.bitwise_and(imgLips, imgColorLips)

        # keeping color lips into original image
        imgColorLips = cv.GaussianBlur(imgColorLips, (7,7), 10)  #adding some blur color so that new lips fix properly
        # convert to gray scale
        imgOriginalGray = cv.cvtColor(imgOriginal, cv.COLOR_BGR2GRAY)
        imgOriginalGray = cv.cvtColor(imgOriginalGray, cv.COLOR_GRAY2BGR)
        imgColorLips = cv.addWeighted(imgOriginalGray, 1, imgColorLips, 0.4, 0)
        cv.imshow("BGR", imgColorLips)


        cv.imshow("Lips", imgLips)
        cv.imshow('Original', imgOriginal)
        if cv.waitKey(0) & 0xFF == ord('q'):
            break


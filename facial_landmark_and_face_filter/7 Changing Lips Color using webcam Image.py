
# Landmark of the face


# Loading necessary library
import numpy as np
import cv2 as cv
import dlib

img = cv.imread("taylor swift.jpg")
img = cv.resize(img, (0,0), None, 0.5,0.5)
imgOriginal = img.copy()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = detector(imgGray)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    imgOriginal = cv.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    landmarks = predictor(imgGray, face)
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv.circle(imgOriginal, (x,y), 5, (50,50,255), cv.FILLED)
        cv.putText(imgOriginal, str(n), (x, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)


    cv.imshow('Original', imgOriginal)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break


# Loading necessary dictionay
import cv2 as cv

#### Capturing video
#cap = cv.VideoCapture('boat.jpeg')
cap = cv.VideoCapture('object track.mp4')

# Declaring Background subtractor
backSub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
#backSub = cv.createBackgroundSubtractorKNN()

# Reading capturing imge/video
while True:
    ret, frame = cap.read()
    if frame is None:
        break

    mask = backSub.apply(frame)

    cv.imshow("Image", mask)
    cv.imshow("Original", frame)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break

# Loading necessary library
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# Defining Tracker
tracker = cv.TrackerMOSSE_create()
tracker = cv.TrackerCSRT_create()

# Taking image
success, img = cap.read()
bbox = cv.selectROI("Tracking", img, False)
tracker.init(img, bbox)

# Drawing bounding box
def drawBox(img, bbox):
    x,y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle(img, (x,y), ((x+w), (y+h)), (255,0,255), 3,1)
    cv.putText(img, "Tracking", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# Opening webcam of video tracking
while True:
    timer = cv.getTickCount() # declare timer
    success, img = cap.read()

    #
    success, bbox = tracker.update(img)
    if success:
        drawBox(img, bbox)
    else:
        cv.putText(img, "Lost", (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv.getTickFrequency()/(cv.getTickCount() - timer)
    cv.putText(img, str(int(fps)), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


    cv.imshow("Tracking", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

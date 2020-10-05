import cv2 as cv
import numpy as np

# reading image
img = cv.imread("/home/hasan/Downloads/hasan.png")

# creating function for trackbar
def empty(a):
    pass

# creating trackbar
cv.namedWindow('trackbar')
cv.resizeWindow("trackbar", 640, 240)
cv.createTrackbar('Hue Min', 'trackbar', 0,179, empty)
cv.createTrackbar('Hue Max', 'trackbar', 179,179, empty)
cv.createTrackbar('Sat Min', 'trackbar', 0,255, empty)
cv.createTrackbar('Sat Max', 'trackbar', 105,255, empty)
cv.createTrackbar('Val Min', 'trackbar', 0,255, empty)
cv.createTrackbar('Val Max', 'trackbar', 155,255, empty)

while True:
    img = cv.imread("/home/hasan/Downloads/hasan.png")
    # convert to gray scale
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # accessing value of trckbar
    h_min = cv.getTrackbarPos("Hue Min", 'trackbar')
    h_max = cv.getTrackbarPos("Hue Max", 'trackbar')
    s_min = cv.getTrackbarPos("Sat Min", 'trackbar')
    s_max = cv.getTrackbarPos("Sat Max", 'trackbar')
    v_min = cv.getTrackbarPos("Val Min", 'trackbar')
    v_max = cv.getTrackbarPos("Val Max", 'trackbar')

    #printing value
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    # making mask image
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, s_max])
    mask = cv.inRange(img_hsv, lower, upper)

    # taking a mask image
    img_result = cv.bitwise_and(img, img_hsv, mask=mask)

    cv.imshow('Original_image', img)
    cv.imshow('Gray_imge', img_hsv)
    cv.imshow('mask_imge', mask)
    cv.imshow('result_imgae_mask', img_result)
    cv.waitKey(1)




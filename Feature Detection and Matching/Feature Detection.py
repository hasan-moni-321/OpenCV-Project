# Loading necessary library
import cv2 as cv
import numpy as np

# Reading images
img1 = cv.imread("venom1.jpeg")
img2 = cv.imread("venom2.jpeg")

# loading orb algorithm
orb = cv.ORB_create()

# detecting and computing of features
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# drawing keypoints
imgkp1 = cv.drawKeypoints(img1, kp1, None)
imgkp2 = cv.drawKeypoints(img2, kp2, None)

# visualize image
cv.imshow("img1", img1)
cv.imshow("img2", img2)
cv.imshow('imgkp1', imgkp1)
cv.imshow('imgkp2', imgkp2)


cv.waitKey(0)


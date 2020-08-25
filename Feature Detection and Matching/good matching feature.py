import cv2 as cv
import numpy as np

img1 = cv.imread("venom1.jpeg", 0)
img2 = cv.imread("venom2.jpeg", 0)

# loading orb algorithm for detecting features
orb = cv.ORB_create(nfeatures=1000)

# detecting and computing of features
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# drawing keypoints
imgkp1 = cv.drawKeypoints(img1, kp1, None)
imgkp2 = cv.drawKeypoints(img2, kp2, None)

# Finding matching
bf = cv.BFMatcher()
match = bf.knnMatch(des1, des2, k=2)

# Taking only good matching
good_match = []
for m,n in match:
    if m.distance <0.75 * n.distance:
        good_match.append([m])
        
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_match, None, flags=2)

# Total number of good matchs
print("Total number of good matchs are : ", len(good_match))
print("Shape of the des1 is :", des1.shape)

# visualize image
#cv.imshow("img1", img1)
#cv.imshow("img2", img2)
cv.imshow('imgkp1', imgkp1)
cv.imshow('imgkp2', imgkp2)
cv.imshow('imgkp3', img3)


cv.waitKey(0)


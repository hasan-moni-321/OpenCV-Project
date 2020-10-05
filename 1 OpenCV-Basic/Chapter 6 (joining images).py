import cv2 as cv
import numpy as np

# reading image
whale = cv.imread("/home/hasan/Downloads/whale.jpg")
hasan = cv.imread("/home/hasan/Downloads/hasan.png")

# horizontal joining two same images
whale_horizontal = np.hstack((whale, whale, whale))

# vertically joining two same images
whale_vertical = np.vstack((whale, whale))


cv.imshow('Horizontal', whale_horizontal)
cv.imshow('Vertical', whale_vertical)
cv.waitKey(0)





import cv2 as cv
import numpy as np

# reading image
img = cv.imread("/home/hasan/Downloads/card.webp")

# selecting position
width, height = 250, 350

points1 = np.float32([[111,219], [287,188], [154,482], [352, 440]])
points2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])

matrix = cv.getPerspectiveTransform(points1, points2)
imgout = cv.warpPerspective(img, matrix, (width, height))

print('Shape of the original iamge is :', img.shape)
cv.imshow("Image", img)
cv.imshow("output", imgout)
cv.waitKey(0)






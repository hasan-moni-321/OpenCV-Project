import cv2 as cv
import numpy as np


# reading image
whale = cv.imread("/home/hasan/Downloads/whale.jpg")
hasan = cv.imread("/home/hasan/Downloads/hasan.png")

# creating function
def stack_image(image1,image2, image3, T):
    if T==True:
        stack_img = np.hstack((image1, image2, image3))
        # calling function
        stack_img = stack_more_image(stack_img, stack_img)

    else:
        stack_img = np.vstack((image1, image2, image3))

    return stack_img


# function for more stack
def stack_more_image(image1, image2):
    stack = np.vstack((image1,image2))
    return stack


# calling function
stack_img = stack_image(whale, whale, whale, True)


cv.imshow('Horizontal', stack_img)
cv.waitKey(0)

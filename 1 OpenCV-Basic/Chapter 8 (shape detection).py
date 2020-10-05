import cv2 as cv
import numpy as np


# reading image
img = cv.imread("/home/hasan/Downloads/shape4.jpg")

# converting to gray scale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# converting to gaussian blue
img_blur = cv.GaussianBlur(img_gray, (7,7), 1)
# converting to canny image
img_canny = cv.Canny(img_blur, 50, 50)


#
def get_contours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2:]

    # finding area of each figure
    for cnt in contours:
        area = cv.contourArea(cnt)
        print('Area of the diagram is :', area)

        #drawing every area in figure
        take_image = np.zeros_like(img)
        a = cv.drawContours(take_image, cnt, -1, (255,0,0), 3)
        cv.imshow('contour image', a)

        # printing length of the contour
        length = cv.arcLength(cnt, True)
        print('length of the contour is :', length)

        # printing corner of the diagram
        corner = cv.approxPolyDP(cnt, 0.02*length, True)
        print(len(corner))

        objCor = len(corner)
        # making rectangle in every figure
        x,y, w, h = cv.boundingRect(corner)

        # printing name of figure
        if objCor == 3: objectType="Tri"
        elif objCor ==4:
            aspRatio = w/float(h)
            if aspRatio > 0.95 and aspRatio <1.05: objectType='Square'
            else: objectType='Rectangle'

        elif objCor>4: objectType="Circles"
        else: objectType="None"

        # square around every diagram
        cv.rectangle(a, (x,y), (x+w, y+h), (0,255,0), 2)

        #keeping text to diagram
        cv.putText(a, objectType,
                  (x+(w//2)-10, y+(h//2)-10), cv.FONT_HERSHEY_COMPLEX, 0.5,
                  (0,255,255), 2)



#calling get_contours function
get_contours(img_canny)


cv.imshow("stack image", img_gray)
cv.imshow("stack image", img_blur)
cv.imshow("stack image", img_canny)
cv.waitKey(0)



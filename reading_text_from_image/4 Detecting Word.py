
# Detecting words

import cv2 as cv
import pytesseract



pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"

img = cv.imread("text detection.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#Reading text from the image
print("Text of the image :", pytesseract.image_to_string(img))
# reading location of the character from image
print("Character location of the image :", pytesseract.image_to_boxes(img))



## Detecting Words
hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_data(img)
#print(boxes)

for x,b in enumerate(boxes.splitlines()):
    if x != 0:
        b = b.split()
        #print(b)
        if len(b)==12:

            # Drawing rectangle around the character
            x,y,w,h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv.rectangle(img, (x,y), (w+x,h+y), (0,0,255), 1)
            cv.putText(img, b[-1], (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 0)


cv.imshow("Result", img)
cv.waitKey(0)


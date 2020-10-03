

# Detecting character and drawing box in the character of the image

import cv2 as cv
import pytesseract


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"

img = cv.imread("text detection.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#Reading text from the image
print("Text of the image :", pytesseract.image_to_string(img))
# reading location of the character from image
print("Character location of the image :", pytesseract.image_to_boxes(img))


## Detecting character
hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    #print(b)
    b = b.split(' ')
    #print(b)
    # Drawing rectangle around the character
    x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv.rectangle(img, (x,hImg-y), (w,hImg-h), (0,0,255), 1)
    cv.putText(img, b[0], (x, hImg-y+25), cv.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 0)


cv.imshow("Result", img)
cv.waitKey(0)


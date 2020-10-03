

import cv2 as cv
import pytesseract


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"

img = cv.imread("text detection.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#Reading text from the image
print("Text of the image :", pytesseract.image_to_string(img))
# reading location of the character from image
print("Character location of the image :", pytesseract.image_to_boxes(img))


cv.imshow("Result", img)
cv.waitKey(0)












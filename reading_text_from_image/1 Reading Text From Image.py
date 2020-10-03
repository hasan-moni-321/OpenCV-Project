
# Reading text from image

import cv2 as cv
import pytesseract


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"

img = cv.imread("text detection.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img))

cv.imshow("Result", img)
cv.waitKey(0)

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading necessary library
import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode


# In[ ]:


# Reading image
img = cv.imread("/home/hasan/Downloads/bar_code 2.png")
code = decode(img)
print('All information of the br code are :', code)


# In[ ]:


# Declare webcam image
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# In[ ]:


# Reading data from My_Data_File
with open("/home/hasan/Downloads/My_Data_File.txt") as f:
    myDataList = f.read().splitlines()
print(myDataList)


# In[ ]:


# Reading webcam image
while True:
    success, img = cap.read()
    for barcode in decode(img):
        print("Only rect information :\n", barcode.rect)
        print("Bite code :\n", barcode.data)

        # decode the data
        my_data = barcode.data.decode('utf-8')
        print("Decoded data :\n", my_data)
        
        # checking new bar code with our given data
        if my_data in myDataList:
            myOutput = "Authorized"
            myColor = (0, 255, 0)
        else:
            myOutput = "Un-authorized"
            myColor = (0, 0, 255)

        # Drawing a rectangle in br code image
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(img, [pts], True, myColor, 5)

        # Writing text to barcode image
        pts2 = barcode.rect
        cv.putText(img, myOutput, (pts2[0], pts2[1]), cv.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)

    cv.imshow('Result', img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


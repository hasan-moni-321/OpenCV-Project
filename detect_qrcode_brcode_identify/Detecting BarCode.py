#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Reading necessary library
import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode


# In[ ]:


# Reading an image
img = cv.imread("/home/hasan/Downloads/bar_code 2.png")
code = decode(img)
print('All information of the br code are :', code)


# In[ ]:


# Declare webcam image
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


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

        # Drawing a rectangle in br code image
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1,1,2))
        cv.polylines(img, [pts], True, (255,255,0), 5)
        
        # Writing text to barcode image
        pts2 = barcode.rect
        cv.putText(img, my_data, (pts2[0], pts2[1]), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)
    
    cv.imshow('Result', img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





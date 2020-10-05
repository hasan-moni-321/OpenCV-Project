
# Loading necessary library
import numpy as np
import cv2 as cv
import face_recognition


# Loading images
img_train = face_recognition.load_image_file("/content/drive/My Drive/Open CV/image/bill gates1.jpeg")
img_test  = face_recognition.load_image_file("/content/drive/My Drive/Open CV/image/bill gates2.jpeg")

# Color to Gray scale
img_train = cv.cvtColor(img_train, cv.COLOR_BGR2RGB)
img_test = cv.cvtColor(img_test, cv.COLOR_BGR2RGB)

# Finding face location and face encoding of train image
face_loc_train = face_recognition.face_locations(img_train)[0]  
face_encode_train = face_recognition.face_encodings(img_test)[0]
cv.rectangle(img_train, (face_loc_train[3], face_loc_train[0]), (face_loc_train[1], face_loc_train[2]), (255,0,0), 2)

# Finding face location and face encoding of train image
face_loc_test = face_recognition.face_locations(img_test)[0]  
face_encode_test = face_recognition.face_encodings(img_test)[0]
cv.rectangle(img_train, (face_loc_test[3], face_loc_test[0]), (face_loc_test[1], face_loc_test[2]), (255,0,0), 2)

# Compare face
compare = face_recognition.compare_faces([face_encode_train], face_encode_test)

# Finding difference of the face
distance = face_recognition.face_distance([face_encode_train], face_encode_test)
print("Comparison result :{} \nDistance result is {}".format(compare, distance))

# text on image
cv.putText(img_test, f'{compare} {round(distance[0], 2)}', (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

# visualize image
cv.imshow('Train image', img_train)
cv.imshow('Test image', img_test)
cv.waitKey(0)








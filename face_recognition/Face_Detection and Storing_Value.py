# Loading necessary library
import numpy as np
import cv2 as cv
import os
import face_recognition
from datetime import datetime

# Reading all the images
path = '/home/hasan/Downloads/image'
image = []
classNames = []
myList = os.listdir(path)
print("Name of the images with format :\n", myList)

for cl in myList:
    curlImage = cv.imread(f'{path}/{cl}')
    image.append(curlImage)
    classNames.append(cl.split('.')[0])

print("Name of the images without format :\n", classNames)


# Encoding images
def ImageEncode(image):
    encodeList = []
    for img in image:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


encodeListKnown = ImageEncode(image)
print("Total number of encoded images :", len(encodeListKnown))


# Saving image name and time to another file
def markAttendance(name):
    with open('/home/hasan/Downloads/Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        # print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


# accessing webcam
cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv.cvtColor(imgs, cv.COLOR_BGR2RGB)

    # finding location and encoding of webcam image
    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFace = face_recognition.face_encodings(imgs, faceCurFrame)

    # Compare and find distance between web_image and given images
    for encodedFace, faceLoc in zip(encodeCurFace, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodedFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodedFace)
        print(faceDis)
        # finding index of most similar
        matchIndex = np.argmin(faceDis)

        # selecting name
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

            # location of the face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # multiplying with 4 because I reduce the wecam pic size in 1/4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv.imshow('webcam', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
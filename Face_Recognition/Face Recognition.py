#!/usr/bin/env python
# coding: utf-8

# In[2]:


import face_recognition
import os
import cv2


# In[16]:


KNOWN_FACES_DIR = '/home/hasan/Downloads/images'
UNKNOWN_FACES_DIR = '/home/hasan/Downloads/images2'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model


# In[17]:


# Returns (R, G, B) from name
def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


# In[18]:


print('Loading known faces...')
known_faces = []
known_names = []


# Reading images from multiple sub folder. Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Reading images one by one
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]

        # appending images and labels 
        known_faces.append(encoding)
        known_names.append(name)
            


# In[ ]:


# Unknown face process
for filename in os.listdir(UNKNOWN_FACES_DIR):

    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')

    # Location of the faces
    locations = face_recognition.face_locations(image, model=MODEL)
    # Encoding of the faces
    encodings = face_recognition.face_encodings(image, locations)

    # Converting image from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

        # Compare face
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')

            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = name_to_color(match)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





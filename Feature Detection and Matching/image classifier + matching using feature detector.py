import cv2 as cv
import os


images = []
class_names = []

path = "/home/hasan/Downloads/Venom"
imgList = os.listdir(path)

for img in imgList:
    image = cv.imread(f'{path}/{img}')
    images.append(image)
    class_names.append(os.path.splitext(img)[0])
print(class_names)

orb = cv.ORB_create(nfeatures=1000)

def find_des(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

def find_best_match(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv.BFMatcher()

    match_list = []
    for des in desList:
        matches = bf.knnMatch(des, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75* n.distance:
                good.append([m])
        match_list.append(len(good))
    print(match_list)


desList = find_des(images)
print(len(desList))

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    ori_img = img.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    find_best_match(img_gray, desList)

    cv.imshow("Ori_Img", ori_img)
    cv.waitKey(1)


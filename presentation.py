import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

folder = "face-recoginition/Images"
images = []
classNames = []
myList = os.listdir(folder)

for cl in myList:
    curImage = cv2.imread(f'{folder}/{cl}')
    images.append(curImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def mark_attends(name):
    with open("face-recoginition/attendancebook.csv", 'r+') as f:
        dataList = f.readlines()
        # print(dataList)
        names_list = []
        for line in dataList:
            entry = line.split(',')
            names_list.append(entry[0])
        if name not in names_list:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name}, {dtString}\n')


encodeKnown = find_encodings(images)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    facescur_frame = face_recognition.face_locations(img_small)
    encode_cur_frame = face_recognition.face_encodings(img_small, facescur_frame)

    for encode_face, face_loc in zip(encode_cur_frame, facescur_frame):
        matches = face_recognition.compare_faces(encodeKnown, encode_face)
        faceDis = face_recognition.face_distance(encodeKnown, encode_face)

        match_index = np.argmin(faceDis)
        if matches[match_index]:
            name = classNames[match_index].upper()

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y1), (0, 0, 255), 3)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            mark_attends(name)

    cv2.imshow("Camera Image", img)
    cv2.waitKey(1)

import cv2
import face_recognition as fr
import os
import numpy as np
from datetime import datetime

images = []
path = "Images"
classes = []
classList = os.listdir(path)
print(classList)
for classs in classList:
    currentImage = cv2.imread(f'{path}/{classs}')
    images.append(currentImage)
    classes.append(os.path.splitext(classs)[0])
print(classes)

def encodings(images):
    encodingsList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = fr.face_encodings(img)[0]
        encodingsList.append(encodes)
    return encodingsList

knownEncodings= encodings(images)
print("The encoding of images process is completed")


def markAttendance(person):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if person not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{person},{dtString}')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgresized = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    facelocCurrentFrame = fr.face_locations(imgresized)
    encodesCurrentFrame = fr.face_encodings(imgresized, facelocCurrentFrame)

    for unknownEncodings, faceLoc in zip(encodesCurrentFrame, facelocCurrentFrame):
        faceMatch = fr.compare_faces(knownEncodings, unknownEncodings)
        faceMatchDis = fr.face_distance(knownEncodings, unknownEncodings)
        print(faceMatchDis)
        faceMatchIndex = np.argmin(faceMatchDis)

        if faceMatch[faceMatchIndex]:
            person = classes[faceMatchIndex]
            print(person)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, person, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendance(person)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
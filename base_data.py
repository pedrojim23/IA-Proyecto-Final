import cv2
import os
import cv2.data
import imutils

subjectName = 'Mari'
dataPath = 'C:/Users/PJimenez/Desktop/IAProy/data'
subjectPath = dataPath + '/' + subjectName
'''-----------------------------------------------------'''
if not os.path.exists(subjectPath):
    print('Carpeta creada: ', subjectPath)
    os.makedirs(subjectPath)
'''-----------------------------------------------------'''

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassification = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = capture.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassification.detectMultiScale(gray, 1.3, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cara = auxFrame[y:y + h, x:x + w]
        cara = cv2.resize(cara, (720,720), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(subjectPath + '/cara_{}.jpg'.format(count),cara)
        count = count + 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k==27 or count>=350:
        break

capture.release()
cv2.destroyAllWindows()
import cv2
import os

dataPath = 'C:/Users/PJimenez/Desktop/IAProy/data'
imgPath = os.listdir(dataPath)
print('imgPath= ', imgPath)

reconocimiento_facial = cv2.face.EigenFaceRecognizer_create()

reconocimiento_facial.read('ModeloFaceFrontalData.xml')

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

caraClassification = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = capture.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    caras = caraClassification.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in caras:
        cara = auxFrame[y:y+h, x:x+w]
        cara = cv2.resize(cara, (720,720), interpolation=cv2.INTER_CUBIC)
        resultado = reconocimiento_facial.predict(cara)

        cv2.putText(frame, '{}'.format(resultado), (x,y-5), 1, 1.3, (255,255,0), 1, cv2.LINE_AA)

        if resultado[1] < 25000:
            cv2.putText(frame, '{}'.format(imgPath[resultado[0]]), (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x,y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows
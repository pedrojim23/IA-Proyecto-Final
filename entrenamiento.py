import cv2
import os
import numpy as np

dataPath = 'C:/Users/PJimenez/Desktop/IAProy/data'
subjectsList = os.listdir(dataPath)
print('Lista personas: ', subjectsList)

labels = []
carasData = []
count = 0

for nombreDireccion in subjectsList:
    subjectPath = dataPath + '/' + nombreDireccion
    print('Viendo imgs')

    for fileNombre in os.listdir(subjectPath):
        print('Caras: ', nombreDireccion + '/' + fileNombre)
        labels.append(count)

        carasData.append(cv2.imread(subjectPath + '/' + fileNombre, 0))
        img = cv2.imread(subjectPath + '/' + fileNombre, 0)

        ''' cv2.imshow('img', img)
        cv2.waitKey(10) '''

    count = count + 1

#cv2.destroyAllWindows()
''' print('labels= ', labels)
print('Numero labels 0: ', np.count_nonzero(np.array(labels)==0))
print('Numero labels 0: ', np.count_nonzero(np.array(labels)==1)) '''

reconocimiento_facial = cv2.face.EigenFaceRecognizer_create()
print('Entrenando...')
reconocimiento_facial.train(carasData, np.array(labels))
reconocimiento_facial.write('ModeloFaceFrontalData.xml')
print("Modelo guardado")

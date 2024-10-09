#####   IMPORTACIONES   #####
import cv2
import os
import shutil
import numpy as np





##### VARIABLES GLOBALES #####

##### SE ASEGURA QUE EXISTAN LOS DIRECTORIOS A UTILIZAR #####
if not os.path.exists('rostros_temporales'):
    os.makedirs('rostros_temporales')
if not os.path.exists('rostros_identificados'):
    os.makedirs('rostros_identificados')
if not os.path.exists('modelos'):
    os.makedirs('modelos')

##### RUTAS A DIRECTORIOS #####
ruta_rostros_temporales = 'rostros_temporales'
ruta_rostros_identificados = 'rostros_identificados'
ruta_modelos = 'modelos'

##### CÁMARAS #####
cam_1 = cv2.VideoCapture(0)

##### RECONOCIMIENTO FACIAL #####
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
reconocedor = cv2.face.LBPHFaceRecognizer_create()





##### FUNCIONES #####

##### CAPTURA #####
def captura():
    c = 1
    while True:
        resultado, frame = cam_1.read()

        if resultado is False:
            break

        frame_aux = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostros = detector.detectMultiScale(frame, 1.2, 5) ### CALIBRAR DETECTOR DE ROSTROS ###

        #c = 1

        for (x, y, w, h) in rostros:
            rostro = frame_aux[y : y + h, x : x + w]
            rostro = cv2.resize(rostro, (150, 150), cv2.INTER_CUBIC)

            cv2.imwrite(ruta_rostros_temporales + '/' + 'rostro_{}.jpg'.format(c), rostro)

            c += 1



##### ENTRENAMIENTO #####
def entrenamiento():
    personas = os.listdir(ruta_rostros_identificados)
    if personas.count('.DS_Store') > 0:
        personas.remove('.DS_Store')

    info_rostros = []
    etiquetas = []
    c = 0

    for persona in personas:
        ruta_rostros = ruta_rostros_identificados + '/' + persona
        rostros = os.listdir(ruta_rostros)
        if rostros.count('.DS_Store') > 0:
            rostros.remove('.DS_Store')
        print('\nLeyendo rostros de: ', persona, '...\n')

        for rostro in rostros:
            print('Rostro analizado: ', persona + '/' + rostro)
            info_rostro = cv2.imread(ruta_rostros + '/' + rostro, 0)
            info_rostros.append(info_rostro)
            etiquetas.append(c)
        print()
        c += 1

    print('Número total de rostros analizados: ', len(etiquetas), '\n')

    print('Entrenando modelo...\n')
    reconocedor.train(info_rostros, np.array(etiquetas))
    ruta_modelo = ruta_modelos + '/' + 'modelo_prueba.xml'
    reconocedor.write(ruta_modelo)
    print('Modelo guardado con éxito!!\n')

    cv2.destroyAllWindows()



##### RECONOCIMIENTO #####
def reconocimiento():
    personas = os.listdir(ruta_rostros_identificados)
    if personas.count('.DS_Store') > 0:
        personas.remove('.DS_Store')

    ruta_modelo = ruta_modelos + '/' + 'modelo_prueba.xml'

    reconocedor.read(ruta_modelo)

    while True:
        resultado, frame = cam_1.read()

        if resultado == False:
            break
        
        frame_aux = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostros = detector.detectMultiScale(frame, 1.2, 5) ### CALIBRAR DETECTOR DE ROSTROS ###

        for (x, y, w, h) in rostros:
            rostro = frame[y : y + h, x : x + w]
            rostro = cv2.resize(rostro, (150, 150), cv2.INTER_CUBIC)
            prediccion = reconocedor.predict(rostro)

            cv2.putText(frame_aux, '{}'.format(prediccion), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if prediccion[1] < 100: ### CALIBRAR RIGUROSIDAD DEL RECONOCEDOR DE ROSTROS ###
                cv2.putText(frame_aux, '{}'.format(personas[prediccion[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame_aux, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame_aux, 'desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame_aux, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Deteccion de Rostros", frame_aux)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break





##### SISTEMA PRINCIPAL #####

#captura()
#entrenamiento()
reconocimiento()

"""
rostros_temporales = os.listdir(ruta_rostros_temporales)
modelos = os.listdir(ruta_modelos)

##### CAMARAS #####
cam_1 = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#face_recognizer = cv2.face.LBPHFaceRecognizer_create()

while True:
    resultado, frame = cam_1.read()

    if resultado is False:
        break

    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_aux = frame_gris.copy()

    rostros = faceClassif.detectMultiScale(frame_gris, 1.2, 5)

    c = 1

    for (x, y, w, h) in rostros:
        rostro = frame_aux[y : y + h, x : x + w]
        rostro = cv2.resize(rostro, (150,150), cv2.INTER_CUBIC)

        cv2.imwrite('rostros_temporales/cam1_{}.jpg'.format(c), rostro)

        persona = 'desconocido'

        modelos = os.listdir(ruta_modelos)

        if len(modelos) > 0:
            for modelo in modelos:
                face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                face_recognizer.read(ruta_modelos + '/' + modelo)
                result = face_recognizer.predict(rostro)

                cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

                if result[1] < 120:
                    persona = modelo.split('.xml')[0]
                    cv2.putText(frame, '{}'.format(rostros_identificados[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    break
                else:
                    cv2.putText(frame, 'desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            if persona != 'desconocido':
                contador = len(os.listdir('rostros_identificados/' + persona)) + 1
                if contador < 101:
                    shutil.move('rostros_temporales/cam1_' + str(c) + '.jpg', 'rostros_identificados/' + persona + '/cam1_' + str(contador) + '.jpg')

                    facesData = []
                    labels = []
                    label = 0

                    for fileName in os.listdir(ruta_rostros_identificados + '/' + persona):
                        labels.append(label)
                        facesData.append(cv2.imread(ruta_rostros_identificados + '/' + persona + '/' + fileName, 0))
                        image = cv2.imread(ruta_rostros_identificados + '/' + persona + '/' + fileName, 0)
                    label = label + 1

                    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                    face_recognizer.train(facesData, np.array(labels))
                    face_recognizer.write('modelos/' + persona + '.xml')
            else:
                contador = len(os.listdir('rostros_identificados')) + 1
                persona = 'persona_' + str(contador)
                os.makedirs('rostros_identificados/' + persona)
                shutil.move('rostros_temporales/cam1_' + str(c) + '.jpg', 'rostros_identificados/' + persona + '/cam1_' + str(contador) + '.jpg')

                facesData = []
                labels = []
                label = 0

                for fileName in os.listdir(ruta_rostros_identificados + '/' + persona):
                    labels.append(label)
                    facesData.append(cv2.imread(ruta_rostros_identificados + '/' + persona + '/' + fileName, 0))
                    image = cv2.imread(ruta_rostros_identificados + '/' + persona + '/' + fileName, 0)
                label = label + 1

                face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                face_recognizer.train(facesData, np.array(labels))
                face_recognizer.write('modelos/' + persona + '.xml')
        
        else:
            contador = len(os.listdir('rostros_identificados')) + 1
            persona = 'persona_' + str(contador)
            os.makedirs('rostros_identificados/' + persona)
            shutil.move('rostros_temporales/cam1_' + str(c) + '.jpg', 'rostros_identificados/' + persona + '/cam1_' + str(contador) + '.jpg')

            facesData = []
            labels = []
            label = 0

            for fileName in os.listdir(ruta_rostros_identificados + '/' + persona):
                labels.append(label)
                facesData.append(cv2.imread(ruta_rostros_identificados + '/' + persona + '/' + fileName, 0))
                image = cv2.imread(ruta_rostros_identificados + '/' + persona + '/' + fileName, 0)
            label = label + 1

            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.train(facesData, np.array(labels))
            face_recognizer.write('modelos/' + persona + '.xml')

        c += 1
    
    cv2.imshow("Deteccion de Rostros", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
"""
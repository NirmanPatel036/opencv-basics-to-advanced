import os
import cv2 as cv
import numpy as np

people = ['Ben Affleck', 'Elton John', 'Jerry Seinfeld', 'Madonna', 'Mindy Kaling']
DIR = r'/Users/nirmanpatel36/Documents/OpenCV/Face Recognition/Faces/'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train(features, labels):
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w] #region of interest
                features.append(faces_roi)
                labels.append(label)

    features = np.array(features, dtype='object')
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)
    face_recognizer.save('face_trained.yml')

    np.save('features.npy', features)
    np.save('labels.npy', labels)

create_train(features, labels)
print('Training done!')

print('The length of features is: ', len(features))
print('The length of labels is: ', len(labels))

face_recognizer = cv.face.LBPHFaceRecognizer_create()
features = np.array(features, dtype='object')
labels = np.array(labels, dtype='int32')

#train the recognizer on the features and labels
face_recognizer.train(features, labels) 

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
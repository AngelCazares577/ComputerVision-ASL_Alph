from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np 
import mediapipe as mp 
from keras.preprocessing.sequence import pad_sequences

db = pickle.load(open("./coordinates.data",'rb'))

data = np.asarray(db['data'])
labels = np.asarray(db['labels'])

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(trainX, trainY)

y_predict = model.predict(testX)

score = accuracy_score(y_predict, testY)

print('{}% samples correctly identified'.format(score * 100))

myFile = open("HandModel",'wb')
pickle.dump({'model':model} , myFile)
myFile.close

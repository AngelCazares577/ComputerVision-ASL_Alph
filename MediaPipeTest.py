
#this is the mediapipe solution for tracking a hands key points, up to 16 I think 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
import pandas as pd
from PIL import Image

from sklearn.preprocessing import LabelBinarizer

#training libraries 


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode = True,)


# Load the dataset
trainingData = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_train.csv')

#extracting labels from data set
labelData = trainingData['label'].values

#Dropping the label column so that we can have only pixel values 
trainingData.drop('label',axis = 1, inplace = True)

#checking if the change went through
#print(train.head())

images = trainingData.values

# Reshape images back to 28x28 and convert from grayscale to RGB
images = np.array([np.reshape(i,28,28) for i in images])


#now to transform our labels
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labelData)

index = 0
# No need to reshape imagesrgb[index] as it's already in the correct shape for RGB images.
plt.imshow(images[index])
plt.show()

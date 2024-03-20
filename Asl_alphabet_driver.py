
#this is the mediapipe solution for tracking a hands key points, up to 16 I think 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
import pandas as pd
from PIL import Image

from sklearn.preprocessing import LabelBinarizer

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
images = np.array([np.reshape(i,(28,28)) for i in images])


#now to transform our labels
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labelData)

mp_hands = mp.solutions.hands

#hands object 
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

for index, img in enumerate(images):
    # Convert grayscale to RGB
    img_rgb = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2RGB)

    # Process the image with mediapipe
    results = hands.process(img_rgb)

    # Check if any hand landmarks were detected
    mp_drawing = mp.solutions.drawing_utils
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1), 
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1)  # styling connections and landmarks
                )

        # Display the image with landmarks
        plt.figure(figsize=(5,5))
        plt.imshow(img_rgb)
        plt.show()
    else:
        print(f"No hands detected in image index {index}")

# close hands
hands.close()

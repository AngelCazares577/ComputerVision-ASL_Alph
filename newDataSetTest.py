import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


#this are general mediapipe hand objects to be utilized 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#created hand object 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#data directory 
DATA_DIR = 'Data'

labels = [] #labels array to contain... labels. 
data = [] #data array to contain.. data.


for directory in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, directory)):
        
        xy_comp = [] #compiles x and y coordinates 

        x_coor = [] #x coordinates of each landmark e.g palm coordinate 
        y_coor = [] #y coordinates of each landmark e.g palm coordinate 

        img = cv2.imread(os.path.join(DATA_DIR, directory, img_path))
        #To avoid issues with a very small amount of images that couldnt load and crashed
        if img is None:
            print(f"Failed to load image at {os.path.join(DATA_DIR, directory, img_path)}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #mediapipe requires RGB colored images 

        mp_drawing = mp.solutions.drawing_utils
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    xy_comp.append(x)
                    xy_comp.append(y)
            labels.append(directory) #in my data folder, each directory is indicative of a letter. 
            data.append(xy_comp)     #append the relevant data for all landsmarks relative to the letter/current directory 

#myFile = open('data.pickle','wb')
#pickle.dump({"data": data,"labels": labels},myFile)
#myFile.close

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

# Now you can use the data object as it was originally created
print(data)
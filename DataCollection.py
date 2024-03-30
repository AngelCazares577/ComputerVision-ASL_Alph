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

EXPECTED_LANDMARKS = 21
for directory in os.listdir(DATA_DIR):
    image_counter = 0
    for img_path in os.listdir(os.path.join(DATA_DIR, directory)):
        if image_counter >= 200:  # Limit to 100 images per directory for this example
            break

        img = cv2.imread(os.path.join(DATA_DIR, directory, img_path))
        if img is None:
            print(f"Failed to load image at {os.path.join(DATA_DIR, directory, img_path)}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            all_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                xy_comp = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))] + \
                          [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]
                all_landmarks.extend(xy_comp)

            # Only proceed if the number of landmarks matches the expected count
            if len(all_landmarks) == EXPECTED_LANDMARKS * 2:  # Multiplied by 2 since xy_comp includes both x and y for each landmark
                labels.append(directory)  # Assuming each directory represents a label
                data.append(all_landmarks)
                image_counter += 1
            else:
                print(f"Skipping image at {os.path.join(DATA_DIR, directory, img_path)} due to unexpected landmark count.")
                
myFile = open('coordinates.data', 'wb')
pickle.dump({"data": data, "labels": labels}, myFile)
myFile.close()
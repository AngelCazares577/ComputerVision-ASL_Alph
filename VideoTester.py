import pickle 
import numpy as np
import mediapipe as mp
import cv2 as cv


#loading model 
myFile = pickle.load(open("HandModel","rb"))
handModel = myFile['model']



stream = cv.VideoCapture(0) #0 is the number for my webcam 

#this are general mediapipe hand objects to be utilized 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#created hand object 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
EXPECTED_LANDMARKS = 21

while True:
    ret,frame = stream.read()
    frame = cv.flip(frame,1)
    frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    xyComp = []
    mp_drawing = mp.solutions.drawing_utils
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=1),  # Custom style for landmarks
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2),  # Custom style for connections
                )

            for hand_landmarks in results.multi_hand_landmarks:
                if len(hand_landmarks.landmark) == EXPECTED_LANDMARKS:
                    for i in range(len(hand_landmarks.landmark)):
                         xyComp.append(hand_landmarks.landmark[i].x)
                         xyComp.append(hand_landmarks.landmark[i].y)
                
                # Proceed with prediction only if xyComp contains coordinates for all expected landmarks
                if len(xyComp) == EXPECTED_LANDMARKS * 2:
                    letter = handModel.predict([np.asarray(xyComp)])
                    predictedLetter = letter[0]  # Assuming the model predicts the label directly
                    print(predictedLetter)
                else:
                    print("Incomplete landmarks detected, skipping frame.")

    cv.imshow('frame', frame)
    if cv.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break


    cv.imshow('frame', frame)
    cv.waitKey(1)


stream.release()
cv.destroyAllWindows()
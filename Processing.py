from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np 
import mediapipe as mp 

db = pickle.load(open("coor.data",'rb'))

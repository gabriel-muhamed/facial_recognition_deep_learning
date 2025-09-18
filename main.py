import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

POS_PATH = os.path.join('data', 'positive')
ANC_PATH = os.path.join('data', 'anchor')
NEG_PATH = os.path.join('data', 'negative')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    cv2.imshow('Image Collection', frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
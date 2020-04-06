import os 
import numpy as np 
import cv2 
import keras
import matplotlib.pyplot as plt
from keras.models import load_model

a = cv2.imread('./fingers/test/4/0a80ac03-5e16-4f79-8282-eba2abe1096e_4R.png')


model = load_model('a1.model')
print("--"*40)
print("--"*40)
print("predicted value is: ", np.argmax(model.predict([a.reshape(1, 128, 128, 3)])))


plt.imshow(a)
plt.show()
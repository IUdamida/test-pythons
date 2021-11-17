#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
val_path="Fruit and Vegetables/validation/"
train_path="Fruit and Vegetables/train/"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_path,
                                                               seed=2509,
                                                               image_size=(224, 224),
                                                              batch_size=32)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(val_path,
                                                              seed=2509,
                                                              image_size=(224, 224),
                                                              shuffle=False,
                                                              batch_size=32)
class_names = train_dataset.class_names
print(class_names)
from tensorflow.keras.models import load_model
new_model=load_model('Models/EfficientNetB3-fruits-98.24.h5')

cap=cv2.VideoCapture(0)

img_counter=0
while True:
    ret, frame= cap.read() #ret para malaman kung nacapture then frame yung mismo sa camera
    
    cv2.imshow('frame',frame)
    
    
    # eto yung pipindutin yung q para malaman na magquit na
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('s'):
        img_name= "screenshot/try_pic_{}.png".format(img_counter)
        cv2.imwrite(img_name,frame)
        print("screenshot taken")
        img_counter+=1
        
        
cap.release()
cv2.destroyAllWindows()

 from keras_preprocessing import image
dir_path= 'screenshot'

for i in os.listdir(dir_path):
    img= image.load_img(dir_path+'//'+ i, target_size=(224,224))
    plt.imshow(img)
    plt.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
pred = new_model.predict(images, batch_size=32)
label = np.argmax(pred, axis = 1)
predict = class_names[np.argmax(pred)]
print("Predicted: "+class_names[np.argmax(pred)])


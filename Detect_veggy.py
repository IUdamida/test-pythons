#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.models import load_model
new_model=load_model('Models/EfficientNetB3-fruits-98.24.h5')
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


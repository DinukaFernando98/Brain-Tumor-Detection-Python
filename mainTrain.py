import cv2
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import  numpy as np
from PIL import Image as Img
from matplotlib import image as img
from keras.utils.np_utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.utils import to_categorical

image_directory='datasets/'

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')

INPUT_SIZE=64
dataset = []
label = []

#print(no_tumor_images)

for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        img=cv2.imread(image_directory+'no/'+image_name)
        img=Img.fromarray(img, 'RGB')
        img=img.resize((64,64))
        dataset.append(np.array(img))
        label.append(0)
        
        
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        img=cv2.imread(image_directory+'yes/'+image_name)
        img=Img.fromarray(img, 'RGB')
        img=img.resize((64,64))
        dataset.append(np.array(img))
        label.append(1)
        
#print(len(dataset))
#print(len(label))

dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)


#Building Model
model = Sequential()


model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3) ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform' ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform' ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
#model.add(Activation('sigmoid')) -- used with binary crossentropy
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=22, 
          validation_data=(x_test, y_test), shuffle=False)

model.save('BrainTumorCategorical22Epochs.h5')
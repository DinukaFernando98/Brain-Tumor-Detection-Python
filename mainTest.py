import cv2
from keras.models import load_model
from PIL import Image as Img
import numpy as np
from torch import tensor

model = load_model('BrainTumorModel.h5')\
    
image=cv2.imread('C:\\Users\\Dilan\\Desktop\\Brain Tumor Detection - RP\\pred\\pred0.jpg')

img=Img.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

#print(img)

img=np.expand_dims(img, axis=0)

#result=np.argmax(img,axis=1)
result=model.predict(img) 
#result=np.argmax(result,axis=1)

#input_img=np.expand_dims(img, axis=0)

#result=model.predict_classes(img)

print(result)
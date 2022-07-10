import  os
from unittest import result
from imageio import save
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('BrainTumorModel.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo==0.:
        print("Not cancerous")
        final_result = "Not cancerous"
        return final_result
    elif classNo==1.:
        print("Cancerous")
        
        final_result = "Cancerous! Immediate consultation recommended"

        final_result2 = "\n\n                                                    Possible Brain Tumor Symptoms includes, Headaches.Seizures or convulsions.Difficulty thinking, speaking or finding words.Personality or behavior changes.Weakness, numbness or paralysis in one part or one side of the body.Loss of balance, dizziness or unsteadiness.Loss of hearing.Vision changes."
        return final_result + final_result2
    else:
        print("No Result")
        return "No Result"
    
def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64,64))
    image = np.array(image)
    image=np.expand_dims(image, axis=0)
    result=model.predict(image) 
    return result

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value)
        return result

if __name__ == '__main__':
    app.run(debug=True)
    
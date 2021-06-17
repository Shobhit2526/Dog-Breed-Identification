#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
try:  
    from PIL import Image
except ImportError:  
    import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import cv2
import requests
from io import BytesIO
warnings.filterwarnings('ignore')
model = tf.keras.models.load_model('model.hdf5')
    
def breed_detection(path):  
    #Preprocessing image
    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    img = img.reshape(1,224,224,3)
    numpydata = np.array(img)
    
    #Predicting the diseases
    predictions= model.predict(numpydata)
    result = predictions.flatten()
    num_list = result.tolist()
    max_value = max(num_list)
    max_index = num_list.index(max_value)
    if max_index == 0:
        ans = 'beagle'
    elif max_index == 1:
        ans='chihuahua'
    elif max_index == 2:
        ans='doberman'
    elif max_index == 3:
        ans='french_bulldog'
    elif max_index == 4:
        ans='golden_retriever'
    elif max_index == 5:
        ans='malamute'
    elif max_index == 6:
        ans='pug'
    elif max_index == 7:
        ans='saint_bernard'
    elif max_index == 8:
        ans='scottish_deerhound'
    else:
        ans='tibetan_mastiff'
    return ans
    
    


# In[3]:


import os  
from flask import Flask, render_template, request


# define a folder to store and later serve the images
UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','jfif'])

app = Flask(__name__)

# function to check the file extension
def allowed_file(filename):  
    return '.' in filename and            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# route and function to handle the home page
# @app.route('/')
# def home_page():  
#     return render_template('index.html')

# route and function to handle the upload page
@app.route('/', methods=['GET', 'POST'])
def upload_page():
    global breed
    breed = ''
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        # if no file is selected
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')
        
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))

        if file and allowed_file(file.filename):

            
            img_src=UPLOAD_FOLDER + file.filename
            breed = breed_detection(img_src)

           
            return render_template('upload.html',
                                   msg='Successfully processed:\n'+file.filename,
                                   breed=breed,
                                   img_src=UPLOAD_FOLDER + file.filename)
    elif request.method == 'GET':
        return render_template('upload.html')
if __name__ == '__main__':  
    app.run(debug = True)
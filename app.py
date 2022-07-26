from cProfile import label
from flask import Flask, render_template, request,jsonify
import numpy as np
import argparse
import time
import cv2
import os
import base64
from base64 import decodebytes
app = Flask(__name__)
import pandas as pd
import os

from keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model.h5')
model.make_predict_function()
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	p = model.predict(i)
	if(p[0][0] > p[0][1]):
		return "error"
	else:
		return "Sreekanth N Kartha"

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Flask app....."

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		#image as base64 encoded string
		img_base64 = request.form['image']
		#convert base64 string to image
		img_data = decodebytes(img_base64.encode('utf-8'))
		img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
		#save image
		cv2.imwrite('image.jpg', img)
		label = predict_label('image.jpg')
        #return label

		return label
	else:
		return "Error"

if __name__ =='__main__':
	app.run(debug=True)
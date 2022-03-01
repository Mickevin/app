from flask import Flask, render_template, request


import numpy as np
import requests
import pickle
import cv2



    
app = Flask(__name__)

def save_pred(path):
    img = cv2.imread(f'data/{path}_leftImg8bit.png')
    plt.imsave('./static/origine.png',img)
    y = model.predict(np.array([img]))[0]
    y = np.array([i.argmax() for i in y])*100
    plt.imsave('./static/pred.png',y.reshape((512,1024)))


@app.route('/', methods=['GET'])
def california_index():
    return render_template("index.html")


@app.route('/predict/', methods=['POST'])
def result():
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt
    
    with open('model_pkl' , 'rb') as f:
        model = pickle.load(f)
    
    if request.method == 'POST':
                 name_img = request.form['name_img']
                 save_pred(name_img)
        
    return render_template("prediction.html", name_img=name_img)

if __name__ == '__main__':
    app.run()

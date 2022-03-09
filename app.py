# Sauvegarde du fichier da# Sauvegarde du fichier data dans l'espace de stockage Azure
from flask import Flask, render_template, request
from PIL.Image import open
from cv2 import resize

app = Flask(__name__)

def load_img_from_azure(name):
    from numpy import array
    from requests import get
    from matplotlib.pyplot import imsave
    # Connection à l'espace de travail d'Azure
    url = f'https://ocia0932039034.blob.core.windows.net/azureml-blobstore-f8554f92-a33d-430c-a1ff-4d9a166c55fc/UI/data/{name}_leftImg8bit.png'
    X = array(open(get(url, stream=True).raw))
    X = resize(X, (1024, 512))
    imsave('./static/origine.png',X)
    return X

@app.route('/', methods=['GET'])
def california_index():
    return render_template("index.html")

@app.route('/predict/', methods=['POST'])
def result():
    from numpy import array
    from tensorflow.keras.models import load_model
    from matplotlib.pyplot import imsave
    model = load_model('./model_cnn/', compile=False)

    if request.method == 'POST':
        name_img = request.form['name_img']
        X = array([load_img_from_azure(name_img)])
        y = array([[i.argmax() for i in u] for u in model.predict(X[:])[0]])
        imsave('./static/pred.png',y)
    return render_template("prediction.html", name_img=name_img)

if __name__ == '__main__':
    app.run()

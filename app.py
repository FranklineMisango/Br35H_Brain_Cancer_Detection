from io import BytesIO
import torch
import numpy as np
import tensorflow as tf
from torch import argmax
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from flask import Flask, jsonify, request

app = Flask(__name__)
LABELS = ['Tumor Less', 'Meningioma', 'Glioma', 'Pituitary']

# Load the saved model
model = tf.keras.models.load_model('/home/misango/code/Br35H_Brain_Cancer_Detection/Models/brain_tumor_classification_efnB7_test10091335.hdf5')

def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_prediction(image_bytes):
    img = preprocess_image(image_bytes)
    predictions = model.predict(img)
    class_id = np.argmax(predictions, axis=1)[0]
    class_name = LABELS[class_id]
    return str(class_id), class_name

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run()

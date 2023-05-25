from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename


#The Trained Models Paths
vgg16_model_path = "/home/misango/code/Br35H_Brain_Cancer_Detection/Models/brain_tumor_classification_VGG16_test1309.h5"
efficientnet_model_path = "/home/misango/code/Br35H_Brain_Cancer_Detection/Models/brain_tumor_classification_efnB7_test10091335.hdf5"

#The models Being Loaded using Tensorflow
vgg16_model = tf.keras.models.load_model(vgg16_model_path, compile=False)
efficientnet_model = tf.keras.models.load_model(efficientnet_model_path, compile=False)

def preprocess_image(file):
    img = Image.open(file)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
app = Flask(__name__)
# Set the path to the folder where uploaded files will be saved
app.config["UPLOAD_FOLDER"] = '/home/misango/code/Br35H_Brain_Cancer_Detection/static/images'
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = preprocess_image(file)

            model_choice = request.form.get("model_choice")
            '''
            if model_choice == "VGG16":
                model = vgg16_model
            else:
            '''
            model = efficientnet_model

            prediction = model.predict(image)
            class_index = np.argmax(prediction)
            classes = ["Glioma", "Meningioma", "Pituitary", "Tumorless"]
            result = classes[class_index]

            # Save the uploaded image file
            image_file = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], image_file))

            return render_template("result.html", result=result, image_file=image_file)

    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import tensorflow as tf

#The model paths
vgg16_model_path = "/home/misango/code/Br35H_Brain_Cancer_Detection/Models/brain_tumor_classification_VGG16_test1309.h5"
efficientnet_model_path = "/home/misango/code/Br35H_Brain_Cancer_Detection/Models/brain_tumor_classification_efnB7_test10091335.hdf5"

# Load your VGG16 model
vgg16_model = tf.keras.models.load_model(vgg16_model_path, compile = False)
efficientnet_model = tf.keras.models.load_model(efficientnet_model_path, compile = False)

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize the image according to the model's input shape
    img = np.array(img) / 255.0  # Normalize pixel values between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def main():
    st.title("BRH35 Lite Classification")
    st.write("Upload an image and select the model to classify the brain tumor.")

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = preprocess_image(uploaded_image)

        model_choice = st.radio("Select the model", ("EfficientNetB7(Recommended Baseline)", "VGG16"))

        if st.button("Classify"):
            st.write("Classifying...")

            if model_choice == "VGG16":
                model = vgg16_model
            else:
                model = efficientnet_model

            prediction = model.predict(image)
            class_index = np.argmax(prediction)
            classes = ["Glioma", "Meningioma", "Pituitary", "Tumorless"]
            result = classes[class_index]

            st.write("Prediction:", result)
            st.image(image[0], use_column_width=True)


if __name__ == "__main__":
    main()

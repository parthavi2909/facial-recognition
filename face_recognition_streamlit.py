import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing import image
import pandas as pd
from tensorflow import keras
import os



def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


col1, col2 = st.columns([1,1])
with col1:
    st.image("woxsen_logo.png")

with col2:
    st.image("appstek_logo.jpg")


def header1(url): 
    st.markdown(f'<p style="color:#800080;font-size:40px;"><strong>{url}</strong></p>', unsafe_allow_html=True)


def predict_image(path):

    img = image.load_img(path, target_size=(224,224,3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    model = keras.models.load_model('face_recognition.h5')
    
    df = pd.read_csv("classes.csv")
    classes = df["class names"]
    
    pred = model.predict(images, batch_size=32)
    header1("Predicted as "+classes[np.argmax(pred)])

def predict():
    st.title("Face Recognition")
   
    uploaded_file = st.file_uploader("Upload the image to Predict", type = ["jfif", "jpeg", "jpg", "png"])

    if uploaded_file:

        image = Image.open(uploaded_file)
        image.save("test_{}".format(uploaded_file.name))
        path = os.path.abspath(os.getcwd()) + "\\test_{}".format(uploaded_file.name)
        #st.image(uploaded_file)
        col1, col2, col3 = st.columns([1,1,1])


        with col2:
            st.image(uploaded_file)
        
    
        col1, col2, col3 = st.columns([1,6,1])


        with col2:
            predict_image(path)
        


        
            


        
if __name__ == "__main__":
    predict()



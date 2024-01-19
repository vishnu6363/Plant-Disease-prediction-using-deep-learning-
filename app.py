# -- coding: utf-8 --
"""
Created on Mon Jan 9 22:09:42 2023
@author: vishn
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://miro.medium.com/max/828/1*mUwYl3gW61G7pafyFnaCvw.gif");
background-size: 180%;
background-position: top;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('./mangomodel.h5')
    return model

def preprocess_image(image):
    # Convert image to NumPy array
    image = np.array(image)
    # Resize image
    image = tf.image.resize(image, (128, 128))
    # Normalize pixel values
    image = image / 255.0
    # Expand dimensions for model input
    image = np.expand_dims(image, axis=0)
    return image

def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    #image = image.img_to_array(image)
   # image=cv2.imread(image)
    image = np.expand_dims(image, axis = 0)
    prediction = model.predict(image)
    return prediction

model = load_model()
st.title('Mongo Plant Disease Prediction')

file = st.file_uploader("Upload an image of a Mango Leaf")

if file is None:
    st.text('Waiting for upload....')
else:
    slot = st.empty()
    slot.text('Running inference....')

    # Load and preprocess the image
    test_image =  tf.keras.preprocessing.image.load_img(file,target_size=(128,128))
     

    st.image(test_image, caption="Input Image", width=150)

    pred = predict_class(test_image, model)
    print(pred)

    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy',
                   'Powdery Mildew', 'Sooty Mould']

    # Get the class index with the highest probability
    predicted_class_index = np.argmax(pred)
    # Get the corresponding class name
    result = class_names[predicted_class_index]

    output = 'The Disease is ' + result

    slot.text('Done')

    st.success(output)

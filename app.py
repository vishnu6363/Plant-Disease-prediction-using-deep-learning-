# -*- coding: utf-8 -*-
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

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('convolutional.h5')
    return model

def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

model = load_model()
st.title('Mongo Plant Disease Prediction')

file = st.file_uploader("Upload an image of a Mongo Leaf")

if file is None:
    st.text('Waiting for upload....')

else:
    slot = st.empty()
    slot.text('Running inference....')

    # Load and preprocess the image
    test_image = Image.open(file).convert("RGB")
    test_image = test_image.resize((128, 128))  # Resize to match the model's input size
    test_image_array = np.array(test_image) / 255.0  # Normalize pixel values to the range [0, 1]

    st.image(test_image, caption="Input Image", width=200)

    pred = predict_class(test_image_array, model)

    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy',
                   'Powdery Mildew', 'Sooty Mould']

    result = class_names[np.argmax(pred)]

    output = 'The Disease is ' + result

    slot.text('Done')

    st.success(output)

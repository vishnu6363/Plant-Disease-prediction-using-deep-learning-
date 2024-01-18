# -*- coding: utf-8 -*-
"""
Created on Mon Jan 9 22:09:42 2023

@author: vishn
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
 
 
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://miro.medium.com/max/828/1*mUwYl3gW61G7pafyFnaCvw.gif");
    background-size: 180%;
    background-position: top;
    background-repeat: no-repeat;
    background-attachment: local;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""
 

st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('convolutional.h5')
    return model

def predict_class(input_image, model):
  img = image.load_img(image_path)
  img = image.img_to_array(img)
  img = tf.image.resize(img, (128, 128)) 
  direct_image = np.expand_dims(img, axis=0) 
  prediction = model.predict(direct_image) 
  return prediction

def main():
    model = load_model()
    st.title('Mongo Plant Disease Prediction')

    file = st.file_uploader("Upload an image of a Leaf")

    if file is None:
        st.text('Waiting for upload....')
    else:
        slot = st.empty()
        slot.text('Running inference....')

        test_image = Image.open(file)
        st.image(file, caption="Input Image")

        pred = predict_class(test_image, model)

        class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy',
                       'Powdery Mildew', 'Sooty Mould']

        result = class_names[np.argmax(pred, axis=1)]

        output = 'The Disease is ' + result

        slot.text('Done')
        st.success(output)

if __name__ == "__main__":
    main()

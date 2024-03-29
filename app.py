# cat_dog_classifier.py

import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
classifier = load_model('resources/dogcat_model_bak.h5')

# Define class labels
class_labels = ['Cat', 'Dog']

# Streamlit app title
st.title('🐱🆚🐶 Cat or Dog Classifier 📸 Predict Your Pet\'s Identity!')

# Upload image
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img1 = image.load_img(uploaded_image, target_size=(64, 64))
    img = image.img_to_array(img1)
    img = img / 255

    # Make predictions
    img = np.expand_dims(img, axis=0)
    prediction = classifier.predict(img, batch_size=None, steps=1)  # gives all class prob.

    st.subheader('Prediction:')

    if (prediction[:, :] > 0.5):
        st.write(f'It\'s a Dog!')
    else:
        st.write(f'It\'s a Cat!')

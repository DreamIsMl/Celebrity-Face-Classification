import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

model_path = 'F:\\class\\DeepLearning\\celebrity image classification\\downloaded_folder (2)'
MODEL = tf.keras.models.load_model(model_path)

class_name = ['lionel_messi', 'maria_sharapova', 'roger_federer', 'serena_williams', 'virat_kohli']
st.set_page_config(page_title="Celebrity Face Classifier:")

st.title('Celebrity Face Detector')
uploaded_file = st.file_uploader('Choose an image')

IMAGE_SIZE = (255, 255)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    if st.button('Predict'):
        st.write('Classifying..')

        image = image.resize(IMAGE_SIZE)
        img_array = np.array(image)
        img_batch = np.expand_dims(img_array, 0)

        prediction = MODEL.predict(img_batch)
        predicted_class = class_name[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2%}")

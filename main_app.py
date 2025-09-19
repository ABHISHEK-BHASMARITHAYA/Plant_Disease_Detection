# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
model = load_model('plant_disease_model.h5')

# Name of Classes (updated full list)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy"
]

# Setting Title of App
st.title("🌱 Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
submit = st.button('Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(f"Image Shape: {opencv_image.shape}")
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert image to 4 Dimension
        opencv_image = np.expand_dims(opencv_image, axis=0)
        
        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        confidence = np.max(Y_pred) * 100

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

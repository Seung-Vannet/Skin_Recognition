import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# Debugging: Show current directory
st.write("Current Directory:", os.getcwd())

# Paths for model, labels, and home image
MODEL_PATH = "trained_model.h5"
LABELS_PATH = "labels.txt"
HOME_IMG_PATH = "home_img.jpg"

# Check if required files exist
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}! Check your file paths.")
    st.stop()

if not os.path.exists(LABELS_PATH):
    st.error(f"Labels file not found at {LABELS_PATH}! Check your file paths.")
    st.stop()

# Load the trained model and class names
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

try:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f]
    st.success("Labels loaded successfully!")
except Exception as e:
    st.error(f"Failed to load labels: {e}")
    st.stop()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to model input size
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    return input_arr

# Function to predict the class of the image
def predict_image(image):
    input_arr = preprocess_image(image)
    predictions = model.predict(input_arr)
    return predictions  # Return all class probabilities

# App UI
st.title("Skin Cancer Recognition App")
st.write("Upload a skin lesion image, and the app will predict its class with probabilities.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict the class probabilities
    predictions = predict_image(image)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    
    # Display prediction result
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write("**Prediction Probabilities:**")
    
    # Prepare data for the bar chart
    probabilities = predictions[0]  # Get probabilities for all classes
    df = pd.DataFrame({
        "Class": class_names,
        "Probability": probabilities
    })
    
    # Display probabilities as a bar chart
    st.bar_chart(df.set_index("Class"))

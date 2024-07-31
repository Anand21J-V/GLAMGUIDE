# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os


# Define the path to the model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "H_CLOSET_MODEL.h5")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the styles data to map predictions to categories
data_dir = './DATASET'
styles_path = os.path.join(data_dir, 'styles.csv')
# Use the on_bad_lines parameter to handle malformed lines
styles_df = pd.read_csv(styles_path, on_bad_lines='skip')

# Ensure the subCategory column contains unique categories for mapping predictions
unique_subcategories = styles_df['subCategory'].unique()

# Function to load and preprocess image
def load_image(image_path, target_size=(120, 160)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Set up the Streamlit app
st.title("Fashion Item Classifier")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # IMAGE UPLOAD KRNE K LIYE
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = load_image(uploaded_file)

    # Predict the subCategory
    pred = model.predict(image)
    subcategory_index = np.argmax(pred, axis=1)[0]

    # Mapping the prediction index to subCategory - TRACK KR LETA HU
    subcategory = unique_subcategories[subcategory_index]
    st.write(f"Predicted SubCategory: {subcategory}")

    # Displaying other attributes
    item_attributes = styles_df[styles_df['subCategory'] == subcategory].iloc[0]
    st.write(f"Season: {item_attributes['season']}")
    st.write(f"Gender: {item_attributes['gender']}")
    st.write(f"MasterCategory: {item_attributes['masterCategory']}")
    st.write(f"ArticleType: {item_attributes['articleType']}")
    st.write(f"BaseColour: {item_attributes['baseColour']}")
    st.write(f"Usage: {item_attributes['usage']}")
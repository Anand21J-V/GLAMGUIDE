from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Define the path to the model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "H_CLOSET_MODEL.h5")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the styles data to map predictions to categories
# 'D:/DATA SCIENCE/CLOSET AI MODEL/DATASET' 
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

@app.route('/classify', methods=['POST'])
def classify_image():
    print("Request received")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file:
        print("File received")
        # Preprocess the image
        image = load_image(file)
        
        # Predict the subCategory
        pred = model.predict(image)
        subcategory_index = np.argmax(pred, axis=1)[0]

        # Mapping the prediction index to subCategory
        subcategory = unique_subcategories[subcategory_index]
        
        # Retrieve other attributes
        item_attributes = styles_df[styles_df['subCategory'] == subcategory].iloc[0]
        response = {
            "Predicted SubCategory": subcategory,
            "Season": item_attributes['season'],
            "Gender": item_attributes['gender'],
            "MasterCategory": item_attributes['masterCategory'],
            "ArticleType": item_attributes['articleType'],
            "BaseColour": item_attributes['baseColour'],
            "Usage": item_attributes['usage']
        }

        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Define the path to the model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/H_CLOSET_MODEL.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define a function to preprocess images for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).resize((120, 160))  # Resize the image
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # Check if image has an alpha channel and remove it
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to decode the model's predictions
def decode_predictions(predictions, feature_labels, class_labels):
    decoded_predictions = {}
    for i, feature in enumerate(feature_labels):
        if i < len(predictions):  # Check if index i is within the bounds of predictions
            predicted_class = np.argmax(predictions[i])
            confidence = predictions[i][predicted_class]
            decoded_predictions[feature] = (class_labels[feature][predicted_class], confidence)
        else:
            decoded_predictions[feature] = ("Unknown", 0.0)  # Default values if prediction is missing
    return decoded_predictions

# List of feature labels in the same order as the model outputs
feature_labels = ['season', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'usage']

# Dictionary of class labels for each feature (should match your training setup)
class_labels = {
    'season': ['Fall', 'Summer', 'Winter'],
    'gender': ['Men', 'Women'],
    'masterCategory': ['Apparel', 'Accessories'],
    'subCategory': ['Topwear', 'Bottomwear', 'Watches'],
    'articleType': ['Shirts', 'Jeans', 'Watches', 'Track Pants', 'Tshirts'],
    'baseColour': ['Navy Blue', 'Blue', 'Silver', 'Black', 'Grey'],
    'usage': ['Casual']
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file.stream).resize((120, 160))  # Resize the image
            img_array = np.array(img)
            if img_array.shape[-1] == 4:  # Check if image has an alpha channel and remove it
                img_array = img_array[..., :3]
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, feature_labels, class_labels)

            # Return predictions to the template
            return render_template('result.html', predictions=decoded_predictions)
    
    return 'Error: No file uploaded.'

if __name__ == '__main__':
    app.run(debug=True)

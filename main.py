import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import os

# interface (frontend)
# let user take pootojrt
# take in image
# print class
# returns a classification (backend)
# put image into model

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load an image for prediction
file_name = input()
image_path = os.path.join(os.getcwd(), file_name)
img = image.load_img(image_path, target_size=(224, 224))  # Load and resize the image
x = image.img_to_array(img)  # Convert the image to a NumPy array
x = np.expand_dims(x, axis=0)  # Add a batch dimension to the array
x = preprocess_input(x)  # Preprocess the input image for the MobileNetV2 model

# Make a prediction with the model
predictions = model.predict(x)  # Predict the probabilities of image classes
decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decode the predicted classes and probabilities

# Print the top predicted classes and probabilities
print('Predictions:')
for class_id, class_name, prob in decoded_predictions:
    print(f'{class_name}: {prob:.2f}')


import os
import tensorflow as tf
import keras_ocr

print(tf.__version__)

# Use EfficientNet from tensorflow.keras.applications
from tensorflow.keras.applications import EfficientNetB0

# Define the path to the images folder
images_folder = 'images'

# Get a list of all image file paths in the folder
images = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

pipeline = keras_ocr.pipeline.Pipeline()

predictions = pipeline.recognize(images)

for image, prediction in zip(images, predictions):
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction)
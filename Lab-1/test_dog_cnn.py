import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = tf.keras.models.load_model("dog_breed_cnn_model.h5")

# the image from google
img_path = "test_labra.webp"
img = image.load_img(img_path, target_size=(512, 512))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Class labels (auto-detect)
class_names = sorted(os.listdir("synthetic_dog_dataset"))

print("Predicted class index:", predicted_class)
print("Predicted breed:", class_names[predicted_class])

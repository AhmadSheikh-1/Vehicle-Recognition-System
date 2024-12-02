import cv2
import numpy as np
import tensorflow as tf

vehicle_model = tf.keras.models.load_model('vehicle_recognition_model.h5')

image = cv2.imread("test_image.jpg")
image = cv2.resize(image, (224, 224))
image = image / 255.0  
image = np.expand_dims(image, axis=0)

output = vehicle_model.predict(image)
color_prediction = np.argmax(output[0]) 
typ_prediction = np.argmax(output[1])   

print("Predicted Color:", color_prediction)
print("Predicted Type:", typ_prediction)

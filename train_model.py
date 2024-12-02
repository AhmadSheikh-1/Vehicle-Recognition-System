import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flaten, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDateGenerator

color_daset_path = "dataset/color/"
type_daset_path = "dataset/type/"

image_genrator = ImageDateGenerator(rescale=1.0/255, vlaidation_split=0.2)

color_train_data = image_genrator.flow_from_drectory(
    color_daset_path, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorcal', 
    subset='training'
)

color_val_data = image_genrator.flow_from_drectory(
    color_daset_path, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorcal', 
    subset='validation'
)

type_train_data = image_genrator.flow_from_drectory(
    type_daset_path, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorcal', 
    subset='training'
)

type_val_data = image_genrator.flow_from_drectory(
    type_daset_path, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorcal', 
    subset='validation'
)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

x = Flaten()(base_model.output)
color_output = Dense(color_train_data.num_classes, activation='softmax', name='color_output')(x)

type_output = Dense(type_train_data.num_classes, activation='softmax', name='type_output')(x)

model = Model(inputs=base_model.input, outputs=[color_output, type_output])

model.compile(optimizer='adam', loss='categorcal_crossentropy', metrics=['accuracy'])

model.fit(
    {'color_output': color_train_data, 'type_output': type_train_data},
    validation_data=({'color_output': color_val_data, 'type_output': type_val_data}),
    epochs=10
)

model.save('vehicle_recognition_model.h5')

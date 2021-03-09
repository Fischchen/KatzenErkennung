import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd

import os
import random
import shutil
import pathlib

from keras_preprocessing.image import ImageDataGenerator

train_imagedatagenerator = ImageDataGenerator(rescale=1/255.0)
validation_imagedatagenerator = ImageDataGenerator(rescale=1/255.0)

train_iterator = train_imagedatagenerator.flow_from_directory(
    "input_for_model/train",
    target_size=(150, 150),
    batch_size=200,
    class_mode="binary")

validation_iterator = validation_imagedatagenerator.flow_from_directory(
    "input_for_model/validation",
    target_size=(150, 150),
    batch_size=50,
    class_mode="binary")



model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_cross_entropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_iterator,
                    validation_data=validation_iterator,
                    steps_per_epoch=100,
                    epochs=50,
                    validation_steps=100)



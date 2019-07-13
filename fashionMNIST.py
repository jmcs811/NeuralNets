from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# helper libs
import matplotlib.pyplot as plt
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

# get the dataset
fashion_mnist = keras.datasets.fashion_mnist

# split data into train and test
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# class names of possible items
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat'
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# image formatting
train_images = train_images / 255.0
test_images = test_images / 255.0

# building the network
model = keras.Sequential([
    # transforms image of 28x28 to 1d array of 784 pix
    keras.layers.Flatten(input_shape=(28, 28)),
    # 128 node, fully connected layer
    keras.layers.Dense(128, activation=tf.nn.relu),
    # output layer, has 10 nodes 1 for each type of item
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)  # run through the data 5 times

# test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# make predictions
predictions = model.predict(test_images)
print(test_labels[0])
print(np.argmax(predictions[0]))

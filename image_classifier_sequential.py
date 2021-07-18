##########################################################
### image_classifier_sequential.py
### Wilson Woods
### 7.9.2021
###
### Image classifier using Keras Sequential API
### From Hands-on Machine Learning by Aurelien Geron
##########################################################

import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# print(X_train_full.shape)
# print(X_train_full.dtype)

# create validation set
# scale pixel intensity to 0-1 range

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] /255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# categories for classifier
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# print(class_names[y_train[0]])


# list of Keras activation functions available
# at https://keras.io/activations/

# classification multi-layer perceptron with 2 hidden layers
# Keras Sequential model
# model = keras.models.Sequential()
# 1st layer, convert each image to 1-D array
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# 2nd layer, Dense hidden layer, 300 neurons
# ReLU activation function
# model.add(keras.layers.Dense(300, activation="relu"))
# 3rd layer, Dense hidden layer, 100 neurons
# ReLU activation function
# model.add(keras.layers.Dense(100, activation="relu"))
# 4th layer, Dense output layer, 10 neurons (one per class)
# softmax activation function
# model.add(keras.layers.Dense(10, activation="softmax"))

# alternate syntax

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

print(model.summary())

# compile the model
model.compile(loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(lr=0.01),
    metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
validation_data=(X_valid, y_valid))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
# set vertical range to [0,1]
plt.gca().set_ylim(0, 1)
plt.show()

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = model.predict_classes(X_new)
np.array(class_names)[y_pred]

y_new = y_test[:3]
print(y_new)
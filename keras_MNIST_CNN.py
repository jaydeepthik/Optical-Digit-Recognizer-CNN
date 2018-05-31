# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:59:55 2018

@author: jaydeep thik
"""

import keras
from keras import models , layers, optimizers, regularizers
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

(X_train ,y_train),(X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000, 28,28,1))
X_test = X_test.reshape((10000, 28,28,1))


X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.

X_tr = X_train[:50000]
X_valid = X_train[50000:]

y_train = to_categorical(y_train, 10)
y_tr = y_train[:50000]
y_valid = y_train[50000:]

y_test = to_categorical(y_test, 10)

network = models.Sequential()
network.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='SAME', input_shape=(28,28,1 )))
network.add(layers.MaxPool2D((2,2)))
network.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='SAME' ))
network.add(layers.MaxPool2D((2,2)))
network.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='SAME' ))
network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer=optimizers.adam(), loss='categorical_crossentropy', metrics=['acc'])
network.fit(X_tr, y_tr, batch_size=64, epochs=5, validation_data=(X_valid, y_valid))
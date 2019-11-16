import json
import os

import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.initializers import Constant
import numpy as np


class Focus:
    def __init__(self):

        small = Constant(value=0.01)

        hacky = Sequential([
            Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=small, bias_initializer='zeros'),
            MaxPool2D(strides=(2, 2), padding='valid'),
            Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=small, bias_initializer='zeros'),
            MaxPool2D(strides=(2, 2), padding='valid'),
            Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=small, bias_initializer='zeros'),
            MaxPool2D(strides=(2, 1), padding='valid'),
            Conv2D(256, (3, 3), padding='valid', activation='relu', kernel_initializer=small, bias_initializer='zeros'),
            MaxPool2D(strides=(2, 1), padding='valid')
        ])

        img = Input(shape=(32, 320, 1))
        self.model = Model(img, hacky(img))

    def create_focus(self, filename):

        if not os.path.exists(filename):
            return -np.ones((42, 76))

        with open(filename) as f:
            meta = json.load(f)

        a = np.zeros((42, 32, 320, 1))

        for index, aap in enumerate(meta):
            x0 = aap['x']
            x1 = x0 + aap['width']
            a[index, :, x0:x1, :] = 1.0

        attention_target = self.model.predict(a)
        b = attention_target.squeeze(axis=1)[:, :, 0]
        c = tf.linalg.normalize(b) * 10.
        d = tf.nn.softmax(c, axis=0)
        return d

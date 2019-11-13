import json
import os

from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.initializers import Constant
import numpy as np


class Focus:
    def __init__(self):

        small = Constant(value=0.001)

        hacky = Sequential([Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=small, bias_initializer='zeros'),
                                 MaxPool2D(strides=(2, 2), padding='valid'),
                                 Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=small, bias_initializer='zeros'),
                                 MaxPool2D(strides=(2, 2), padding='valid'),
                                 Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=small, bias_initializer='zeros'),
                                 MaxPool2D(strides=(2, 1), padding='valid'),
                                 Conv2D(256, (3, 3), padding='valid', activation='relu', kernel_initializer=small, bias_initializer='zeros'),
                                 MaxPool2D(strides=(2, 1), padding='valid')])

        img = Input(shape=(32, 320, 1))
        self.model = Model(img, hacky(img))

    def create_focus(self, filename):

        if not os.path.exists(filename):
            return -np.ones((42, 76))

        with open(filename) as f:
            meta = json.load(f)

        b = np.zeros((42, 76)) + (1 / 76.0)

        a = np.zeros((len(meta), 32, 320, 1))

        for index, aap in enumerate(meta):
            x0 = aap['x']
            x1 = x0 + aap['width']
            a[index, :, x0:x1, :] = 1.0

        blaat = self.model.predict(a)
        for index in range(blaat.shape[0]):
            out = blaat[index]
            out = out.squeeze(axis=0)
            out = out.transpose()
            out = out[0, :]

            if np.max(out) == 0.0:
                out = out + (1 / 76.0)

            out = out / np.sum(out)

            b[index, :] = out

        return b

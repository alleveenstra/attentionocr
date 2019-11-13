import json

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
        b = np.zeros((42, 76)) + (1 / 76.0)

        with open('scripts/train/aaevtqdyfjlhzbrcljgcutq.json') as f:
            meta = json.load(f)

        for index, aap in enumerate(meta):
            a = np.zeros((1, 32, 320, 1))
            x0 = aap['x']
            x1 = x0 + aap['width']
            a[0, :, x0:x1, :] = 1.0

            out = self.model.predict(a)

            out = out.squeeze(axis=0)
            out = out.squeeze(axis=0)
            out = out.transpose()
            out = out[0, :]
            out = out / np.sum(out)

            b[index, :] = out

        return b

        # cv2.imwrite('test01.png', b * 255)




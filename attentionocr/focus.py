import json
import math
import os

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D
import numpy as np

from .model import Encoder


class Focus:
    def create_focus(self, filename):
        if not os.path.exists(filename):
            return -np.ones((42, 76))
        with open(filename) as f:
            meta = json.load(f)
        q = np.zeros((42, 76))
        for index, char in enumerate(meta):
            x0 = char['x']
            x1 = x0 + char['width']
            for layer in Encoder.layers:
                if type(layer) is MaxPool2D:
                    x0 /= float(layer.pool_size[1])
                    x1 /= float(layer.pool_size[1])
                if type(layer) is Conv2D:
                    x0 -= math.ceil(layer.kernel_size[1] / 2.)
                    x1 += math.ceil(layer.kernel_size[1] / 2.)
            x0 = max(math.floor(x0), 0)
            x1 = min(math.ceil(x1), 76)
            q[index, x0:x1] = 10.0
        q = tf.nn.softmax(q, axis=0)
        return q

import json
import math
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D

from .layers import Encoder
from .image import ImageUtil
from .vocabulary import Vocabulary


class Vectorizer:

    def __init__(self, vocabulary: Vocabulary, image_height=32, image_width=320, max_txt_length: int = 42, transform: str = "lowercase"):
        self._vocabulary = vocabulary
        self._max_txt_length = max_txt_length
        self._image_height = image_height
        self._image_width = image_width
        self._encoding_width = Encoder.get_width(image_width)
        self._image_util = ImageUtil(image_height, image_width)
        self._transform = transform

    def load_image(self, image) -> np.ndarray:
        return self._image_util.load(image)

    def create_focus(self, filename) -> tf.Tensor:
        if not os.path.exists(filename):
            return -np.ones((self._max_txt_length, self._encoding_width))
        with open(filename) as f:
            meta = json.load(f)
        q = np.zeros((self._max_txt_length, self._encoding_width))
        for index, char in enumerate(meta):
            x0 = char['x']
            x1 = x0 + char['width']
            for layer in Encoder.layers:
                if type(layer) is MaxPool2D:
                    x0 /= float(layer.pool_size[1])
                    x1 /= float(layer.pool_size[1])
            x0 = max(math.floor(x0), 0)
            x1 = min(math.ceil(x1), self._encoding_width)
            q[index, x0:x1] = 10.0
        q = tf.nn.softmax(q, axis=0)
        return q

    def transform_text(self, target_text: str, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        decoder_input_size = self._max_txt_length if is_training else 1
        decoder_input = np.zeros((decoder_input_size, len(self._vocabulary)), dtype='float32')

        decoder_output_size = self._max_txt_length
        decoder_output = np.zeros((decoder_output_size, len(self._vocabulary)), dtype='float32')

        # transform the text
        if self._transform == "lowercase":
            target_text = target_text.lower()

        # decoder input
        if is_training:
            decoder_input[:, :] = self._vocabulary.one_hot_encode(target_text, decoder_input_size, sos=True, eos=True)
        else:
            decoder_input[:, :] = self._vocabulary.one_hot_encode('', 1, sos=True, eos=False)

        # decoder output
        decoder_output[:, :] = self._vocabulary.one_hot_encode(target_text, decoder_output_size, eos=True)

        return decoder_input, decoder_output

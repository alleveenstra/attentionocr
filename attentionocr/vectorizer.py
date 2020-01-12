from typing import Tuple

import numpy as np

from .image import ImageUtil
from .layers import Encoder
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

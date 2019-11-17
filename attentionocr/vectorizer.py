from typing import List

import numpy as np

from attentionocr.image import ImageUtil
from .vocabulary import Vocabulary


class Vectorizer:

    def __init__(self, vocabulary: Vocabulary, image_height=32, image_width=320, max_txt_length: int = 42, transform: str = "lowercase"):
        self._vocabulary = vocabulary
        self._max_txt_length = max_txt_length
        self._image_height = image_height
        self._image_width = image_width
        self._image_util = ImageUtil(image_height, image_width)
        self._transform = transform

    def load_image(self, image):
        return self._image_util.load(image)

    def transform(self, images: List[np.ndarray], focuses: List[np.ndarray], texts: List[str], is_training: bool = True):
        assert len(images) == len(texts)
        encoder_input = np.zeros((len(texts), self._image_height, self._image_width, 1), dtype='float32')

        decoder_input_size = self._max_txt_length if is_training else 1
        decoder_input = np.zeros((len(texts), decoder_input_size, len(self._vocabulary)), dtype='float32')

        decoder_output_size = self._max_txt_length
        decoder_output = np.zeros((len(texts), decoder_output_size, len(self._vocabulary)), dtype='float32')

        attention_focus = np.zeros((len(texts), decoder_output_size, 77), dtype='float32')


        for sample_index, (image, focus, target_text) in enumerate(zip(images, focuses, texts)):
            # load the image
            encoder_input[sample_index] = image

            attention_focus[sample_index, :, :] = focus

            # transform the text
            if self._transform == "lowercase":
                target_text = target_text.lower()

            # decoder input
            if is_training:
                decoder_input[sample_index, :, :] = self._vocabulary.one_hot_encode(target_text, decoder_input_size, sos=True, eos=True)
            else:
                decoder_input[sample_index, :, :] = self._vocabulary.one_hot_encode('', 1, sos=True, eos=False)

            # decoder output
            decoder_output[sample_index, :, :] = self._vocabulary.one_hot_encode(target_text, decoder_output_size, eos=True)

        return [encoder_input, decoder_input], decoder_output, attention_focus

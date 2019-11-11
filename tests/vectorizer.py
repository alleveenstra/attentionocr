import unittest

from attentionocr import Vocabulary
from attentionocr.vectorizer import Vectorizer
import cv2
import numpy as np


class VectorizerTest(unittest.TestCase):

    def test_image_scaling(self):
        image_height = 32
        image_width = 320
        vec = Vectorizer(['a', 'b'], image_height=image_height, image_width=image_width)
        img = vec.load_image('test_600x100.png')
        cv2.imwrite('out_600x100.png', (img + 1.0) * 127.5)
        assert img.shape == (image_height, image_width, 1)
        img = vec.load_image('test_100x32.png')
        cv2.imwrite('out_100x32.png', (img + 1.0) * 127.5)
        assert img.shape == (image_height, image_width, 1)
        img = vec.load_image('test_50x16.png')
        assert img.shape == (image_height, image_width, 1)
        cv2.imwrite('out_50x16.png', (img + 1.0) * 127.5)
        img = vec.load_image('test_288x32.png')
        assert img.shape == (image_height, image_width, 1)
        cv2.imwrite('out_288x32.png', (img + 1.0) * 127.5)

    def test_too_large_input(self):
        voc = Vocabulary(['a', 'b'])
        vec = Vectorizer(voc, image_height=32, image_width=144, max_txt_length=10)

        inputs, output = vec.transform(['test_100x32.png'], ['aabbaabbaabbaa'])

        assert(output.shape[-1] == len(voc))

        print(np.argmax(inputs[1], axis=-1))
        print(np.argmax(output, axis=-1))

    def test_shapes(self):
        voc = Vocabulary(['a', 'b'])
        vec = Vectorizer(voc, image_height=32, image_width=144, max_txt_length=10)

        inputs, output = vec.transform(['test_100x32.png'], ['aabbaabbaabbaa'], is_training=False)

        assert(inputs[0].shape[0] == 1)
        assert(inputs[0].shape[1] == 32)
        assert(inputs[0].shape[2] == 144)
        assert(inputs[0].shape[3] == 1)

        assert(inputs[1].shape[0] == 1)
        assert(inputs[1].shape[1] == 1)
        assert(inputs[1].shape[2] == len(voc))

        assert(output.shape[0] == 1)
        assert(output.shape[1] >= 10)
        assert(output.shape[2] == len(voc))

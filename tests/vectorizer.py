import unittest

from attentionocr import Vocabulary
from attentionocr.image import ImageUtil
from attentionocr.vectorizer import Vectorizer
import cv2
import numpy as np


class VectorizerTest(unittest.TestCase):

    def test_image_scaling(self):
        image_height = 32
        image_width = 320
        image_util = ImageUtil(image_height=image_height, image_width=image_width)
        img = image_util.load('test_600x100.png')
        cv2.imwrite('out_600x100.png', (img + 1.0) * 127.5)
        self.assertEqual((image_height, image_width, 1), img.shape)
        img = image_util.load('test_100x32.png')
        cv2.imwrite('out_100x32.png', (img + 1.0) * 127.5)
        self.assertEqual((image_height, image_width, 1), img.shape)
        img = image_util.load('test_50x16.png')
        self.assertEqual((image_height, image_width, 1), img.shape)
        cv2.imwrite('out_50x16.png', (img + 1.0) * 127.5)
        img = image_util.load('test_288x32.png')
        self.assertEqual((image_height, image_width, 1), img.shape)
        cv2.imwrite('out_288x32.png', (img + 1.0) * 127.5)

    def test_too_large_input(self):
        voc = Vocabulary(['a', 'b'])
        vec = Vectorizer(voc, image_height=32, image_width=144, max_txt_length=10)

        inputs, output = vec.transform_text(['test_100x32.png'], ['aabbaabbaabbaa'])

        self.assertEqual(len(voc), output.shape[-1])

        print(np.argmax(inputs[1], axis=-1))
        print(np.argmax(output, axis=-1))

    def test_shapes(self):
        voc = Vocabulary(['a', 'b'])
        vec = Vectorizer(voc, image_height=32, image_width=144, max_txt_length=10)

        inputs, output = vec.transform_text(['test_100x32.png'], ['aabbaabbaabbaa'], is_training=False)

        self.assertEqual(1, inputs[0].shape[0])
        self.assertEqual(32, inputs[0].shape[1])
        self.assertEqual(144, inputs[0].shape[2])
        self.assertEqual(1, inputs[0].shape[3])

        self.assertEqual(1, inputs[1].shape[0])
        self.assertEqual(1, inputs[1].shape[1])
        self.assertEqual(len(voc), inputs[1].shape[2])

        self.assertEqual(1, output.shape[0])
        self.assertTrue(output.shape[1] >= 10)
        self.assertEqual(len(voc), output.shape[2])

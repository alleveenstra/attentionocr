import unittest
from attentionocr.vectorizer import VectorizerOCR
import cv2


class VectorizerTest(unittest.TestCase):

    def test_image_scaling(self):
        image_height = 32
        image_width = 320
        vec = VectorizerOCR(['a', 'b'], image_height=image_height, image_width=image_width)
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

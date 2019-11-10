import cv2
import numpy as np


class ImageUtil:

    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

    def load_image(self, filename):
        image = cv2.imread(filename)
        image = self.scale_image(image)
        image = self.grayscale_image(image)
        image = self.pad_image(image)
        image = np.expand_dims(image, axis=2)
        return image

    def pad_image(self, image):
        _, width = image.shape
        if width < self.image_width:
            # zero-pad on the right side
            image = np.pad(image, ((0, 0), (0, self.image_width - width)), 'constant')
        return image

    def grayscale_image(self, image):
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 127.5) - 1.0
        return image

    def scale_image(self, image):
        height, width, _ = image.shape
        scaling_factor = height / self.image_height
        if height != self.image_height:
            if width / scaling_factor <= self.image_width:
                # scale both axis when the scaled width is smaller than the target width
                image = cv2.resize(image, (int(width / scaling_factor), int(height / scaling_factor)), interpolation=cv2.INTER_AREA)
            else:
                # otherwise, compress the horizontal axis
                image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        elif width > self.image_width:
            # the height matches, but the width is longer
            image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        return image

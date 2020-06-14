import cv2
import numpy as np


class ImageUtil:

    def __init__(self, image_height: int, image_width: int):
        self._image_height = image_height
        self._image_width = image_width

    def load(self, filename: str) -> np.ndarray:
        image = cv2.imread(filename)
        image = self.preprocess(image)
        return image

    def preprocess(self, image):
        image = self._scale_axis(image)
        image = self._grayscale(image)
        image = self._pad(image)
        image = np.expand_dims(image, axis=2)
        return image

    def _scale_axis(self, image: np.ndarray) -> np.ndarray:
        height, width, _ = image.shape
        scaling_factor = height / self._image_height
        if height != self._image_height:
            if width / scaling_factor <= self._image_width:
                # scale both axis when the scaled width is smaller than the target width
                image = cv2.resize(image, (int(width / scaling_factor), int(height / scaling_factor)), interpolation=cv2.INTER_AREA)
            else:
                # otherwise, compress the horizontal axis
                image = cv2.resize(image, (self._image_width, self._image_height), interpolation=cv2.INTER_AREA)
        elif width > self._image_width:
            # the height matches, but the width is longer
            image = cv2.resize(image, (self._image_width, self._image_height), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def _grayscale(image: np.ndarray) -> np.ndarray:
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 127.5) - 1.0
        return image

    def _pad(self, image: np.ndarray) -> np.ndarray:
        _, width = image.shape
        if width < self._image_width:
            # zero-pad on the right side
            image = np.pad(image, ((0, 0), (0, self._image_width - width)), 'constant')
        return image

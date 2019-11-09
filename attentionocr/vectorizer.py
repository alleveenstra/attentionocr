import cv2
import numpy as np
from .vocabulary import Vocabulary


class VectorizerOCR:

    def __init__(self, vocabulary: Vocabulary, image_height=32, image_width=320, max_txt_length: int = 40):
        self.vocabulary = vocabulary
        self.max_txt_length = max_txt_length
        self.image_height = image_height
        self.image_width = image_width

    def vectorize(self, filenames: list, texts: list, is_training: bool = True):
        assert len(filenames) == len(texts)
        encoder_input = np.zeros((len(texts), self.image_height, self.image_width, 1), dtype='float32')

        decoder_input_size = self.max_txt_length + 2 if is_training else 1
        decoder_input = np.zeros((len(texts), decoder_input_size, len(self.vocabulary)), dtype='float32')

        decoder_output_size = self.max_txt_length + 2
        decoder_output = np.zeros((len(texts), decoder_output_size, len(self.vocabulary)), dtype='float32')

        for sample_index, (filename, target_text) in enumerate(zip(filenames, texts)):
            # load the image
            encoder_input[sample_index] = self.load_image(filename)

            # decoder input
            if is_training:
                decoder_input[sample_index, :, :] = self.vocabulary.one_hot_encode(target_text, decoder_input_size, sos=True, eos=True)
            else:
                decoder_input[sample_index, :, :] = self.vocabulary.one_hot_encode('', 1, sos=True, eos=False)

            # decoder output
            decoder_output[sample_index, :, :] = self.vocabulary.one_hot_encode(target_text, decoder_output_size, eos=True)

        return [encoder_input, decoder_input], decoder_output

    def load_image(self, filename):
        image = cv2.imread(filename)

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

        image = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 127.5) - 1.0
        _, width = image.shape
        if width < self.image_width:
            # zero-pad on the right side
            image = np.pad(image, ((0, 0), (0, self.image_width - width)), 'constant')

        image = np.expand_dims(image, axis=2)

        return image


class VectorizedBatchGenerator:

    def __init__(self, vectorizer: VectorizerOCR, batch_size: int = 64):
        self.vectorizer = vectorizer
        self.batch_size = batch_size

    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def flow_from_dataset(self, dataset: list, is_training: bool = True):
        current_idx = 0
        batches = list(self.chunks(dataset, self.batch_size))
        while True:
            if current_idx >= len(batches):
                current_idx = 0
            batch = batches[current_idx]
            images, texts = zip(*batch)
            current_idx += 1
            yield self.vectorizer.vectorize(images, texts, is_training)

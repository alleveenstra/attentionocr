import numpy as np

from .image import ImageUtil
from .vocabulary import Vocabulary


class VectorizerOCR:

    def __init__(self, vocabulary: Vocabulary, image_height=32, image_width=320, max_txt_length: int = 40):
        self.vocabulary = vocabulary
        self.max_txt_length = max_txt_length
        self.image_height = image_height
        self.image_width = image_width
        self.image_util = ImageUtil(self.image_height, self.image_width)

    def vectorize(self, filenames: list, texts: list, is_training: bool = True):
        assert len(filenames) == len(texts)
        encoder_input = np.zeros((len(texts), self.image_height, self.image_width, 1), dtype='float32')

        decoder_input_size = self.max_txt_length + 2 if is_training else 1
        decoder_input = np.zeros((len(texts), decoder_input_size, len(self.vocabulary)), dtype='float32')

        decoder_output_size = self.max_txt_length + 2
        decoder_output = np.zeros((len(texts), decoder_output_size, len(self.vocabulary)), dtype='float32')

        for sample_index, (filename, target_text) in enumerate(zip(filenames, texts)):
            # load the image
            encoder_input[sample_index] = self.image_util.load_image(filename)

            # decoder input
            if is_training:
                decoder_input[sample_index, :, :] = self.vocabulary.one_hot_encode(target_text, decoder_input_size, sos=True, eos=True)
            else:
                decoder_input[sample_index, :, :] = self.vocabulary.one_hot_encode('', 1, sos=True, eos=False)

            # decoder output
            decoder_output[sample_index, :, :] = self.vocabulary.one_hot_encode(target_text, decoder_output_size, eos=True)

        return [encoder_input, decoder_input], decoder_output


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

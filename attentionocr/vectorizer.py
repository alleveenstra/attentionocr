import cv2
import numpy as np


class VectorizerOCR:

    def __init__(self, vocabulary: list, image_height=32, image_width=144, max_txt_length: int = 12):
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.target_characters = [self.SOS, self.EOS, self.PAD] + vocabulary
        self.num_decoder_tokens = len(self.target_characters)
        self.character_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
        self.character_reverse_index = dict((i, char) for char, i in self.character_index.items())
        self.max_txt_length = max_txt_length
        self.image_height = image_height
        self.image_width = image_width

    def vectorize(self, filenames: list, texts: list):
        assert len(filenames) == len(texts)
        encoder_input = np.zeros((len(texts), self.image_height, self.image_width, 1), dtype='float32')
        decoder_input = np.zeros((len(texts), self.max_txt_length + 2, self.num_decoder_tokens), dtype='float32')
        decoder_output = np.zeros((len(texts), self.max_txt_length + 2, self.num_decoder_tokens), dtype='float32')

        for sample_index, (filename, target_text) in enumerate(zip(filenames, texts)):
            # Load the image
            encoder_input[sample_index] = self.load_image(filename)

            # ensure the text is not too long
            target_text = target_text[:self.max_txt_length]

            # decoder input
            decoder_input_tokens = [self.SOS] + list(target_text) + [self.EOS]
            for char_pos, char in enumerate(decoder_input_tokens):
                decoder_input[sample_index, char_pos, self.character_index[char]] = 1.
            decoder_input[sample_index, char_pos + 1:, self.character_index[self.PAD]] = 1.

            # decoder output
            decoder_output_tokens = list(target_text) + [self.EOS]
            for char_pos, char in enumerate(decoder_output_tokens):
                decoder_output[sample_index, char_pos, self.character_index[char]] = 1.
            decoder_output[sample_index, char_pos:, self.character_index[self.PAD]] = 1.

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

    def tokens(self):
        return self.target_characters


class VectorizedBatchGenerator:

    def __init__(self, vectorizer: VectorizerOCR, batch_size: int = 64):
        self.vectorizer = vectorizer
        self.batch_size = batch_size

    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def flow_from_dataset(self, dataset: list):
        current_idx = 0
        batches = list(self.chunks(dataset, self.batch_size))
        while True:
            if current_idx >= len(batches):
                current_idx = 0
            batch = batches[current_idx]
            images, texts = zip(*batch)
            current_idx += 1
            yield self.vectorizer.vectorize(images, texts)

import cv2
import numpy as np

class VectorizerOCR:

    def __init__(self, vocabulary: list, max_txt_length: int = 12):
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.target_characters = [self.SOS, self.EOS, self.PAD] + vocabulary
        self.num_decoder_tokens = len(self.target_characters)
        self.character_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
        self.character_reverse_index = dict((i, char) for char, i in self.character_index.items())
        self.max_txt_length = max_txt_length
        self.image_height = 32
        self.image_width = 144

    def vectorize(self, filenames: list, texts: list):
        assert len(filenames) == len(texts)
        encoder_input_data = np.zeros((len(texts), self.image_height, self.image_width, 1), dtype='float32')
        decoder_input_data = np.zeros((len(texts), self.max_txt_length, self.num_decoder_tokens), dtype='float32')
        decoder_target_data = np.zeros((len(texts), self.max_txt_length, self.num_decoder_tokens), dtype='float32')

        for sample_index, (filename, target_text) in enumerate(zip(filenames, texts)):
            # Load the image
            input_image = self.load_image(filename)
            encoder_input_data[sample_index] = input_image

            for char_pos, char in enumerate([self.SOS] + list(target_text) + [self.EOS]):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[sample_index, char_pos, self.character_index[char]] = 1.
                if char_pos > 0:
                    # decoder_target_data will be ahead by one timestep and will not include the start character.
                    decoder_target_data[sample_index, char_pos - 1, self.character_index[char]] = 1.
            decoder_input_data[sample_index, char_pos + 1:, self.character_index[self.PAD]] = 1.
            decoder_target_data[sample_index, char_pos:, self.character_index[self.PAD]] = 1.
        return [encoder_input_data, decoder_input_data], decoder_target_data

    def load_image(self, filename):
        image = cv2.imread(filename)
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 127.5) - 1.0
        image = np.expand_dims(image, axis=2)
        return image

    def tokens(self):
        return self.target_characters


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class VectorizedBatchGenerator:

    def __init__(self, vectorizer: VectorizerOCR, batch_size: int = 64):
        self.vectorizer = vectorizer
        self.batch_size = batch_size

    def flow_from_dataset(self, dataset: list):
        current_idx = 0
        batches = list(chunks(dataset, self.batch_size))
        while True:
            if current_idx >= len(batches):
                current_idx = 0
            batch = batches[current_idx]
            images, texts = zip(*batch)
            current_idx += 1
            yield self.vectorizer.vectorize(images, texts)

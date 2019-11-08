import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Input, Sequential
from tensorflow.python.keras.layers import MaxPool2D, Bidirectional, Concatenate, Conv2D, Activation, RepeatVector, Permute, LSTM, Multiply, BatchNormalization, Flatten, Dense, MaxPooling2D, TimeDistributed, Dot, Softmax, Lambda
from tensorflow.python.keras.models import Model
import tensorflow as tf

import numpy as np

from .vectorizer import VectorizerOCR


class KerasAttentionOCR:

    def __init__(self, vectorizer: VectorizerOCR, units=256, image_height=32, image_width=144):
        self.vectorizer = vectorizer
        self.num_tokens = len(vectorizer.tokens())
        self.max_input_txt_size = self.vectorizer.max_txt_length
        self.max_output_txt_size = self.vectorizer.max_txt_length + 2
        self.image_height = image_height
        self.image_width = image_width

        self.units = units

        # Build the model.
        self.encoder_input = Input(shape=(self.image_height, None, 1))
        self.decoder_input = Input(shape=(None, self.num_tokens))
        self.encoder = Encoder(self.units)
        self.attention = Attention(self.units)
        self.decoder = Decoder(self.units)
        self.output = Output(self.num_tokens)
        self.training_model = self.build_training_model()

        self.training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.training_model.summary()

        self.inference_model = self.build_inference_model()


    def build_training_model(self) -> tf.keras.Model:
        """
        Builds and returns the keras model used during training.

        :return: tf.keras.Model
        """
        encoder_states, state_h, state_c = self.encoder(self.encoder_input)
        initial_state = [state_h, state_c]
        context_vectors, _ = self.attention(self.decoder_input, encoder_states)
        x = tf.concat([self.decoder_input, context_vectors], axis=2)
        decoder_output, _, _ = self.decoder(x, initial_state=initial_state)
        scores = self.output(decoder_output)
        return tf.keras.Model([self.encoder_input, self.decoder_input], scores)

    def build_inference_model(self):
        """
        Builds and returns the keras model used during inference.

        :return: tf.keras.Model
        """
        predictions = []
        prediction = self.decoder_input
        encoder_states, state_h, state_c = self.encoder(self.encoder_input)
        for i in range(self.max_output_txt_size):
            context_vectors, _ = self.attention(prediction, encoder_states)
            x = tf.concat([prediction, context_vectors], axis=2)
            decoder_output, state_h, state_c = self.decoder(
                x, [state_h, state_c])
            prediction = self.output(decoder_output)
            predictions.append(prediction)
        scores = tf.concat(predictions, axis=1)
        return tf.keras.Model([self.encoder_input, self.decoder_input], scores)

    def fit(self, images: list, texts: list, epochs: int = 10, batch_size: int = None, validation_split=0.):
        if batch_size is None:
            batch_size = len(images)
        X, y = self.vectorizer.vectorize(images, texts)
        K.set_learning_phase(1)
        self.training_model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def fit_generator(self, generator, steps_per_epoch: int = None, epochs: int = 1, validation_data=None, validation_steps=None):
        K.set_learning_phase(1)
        self.training_model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_data, validation_steps=validation_steps)

    def predict(self, images):
        K.set_learning_phase(0)
        texts = []
        for image in images:
            # feed the input, retrieve encoder state vectors
            image = np.expand_dims(image, axis=0)
            # states_value = self.inference_encoder.predict(image)

            target_seq = np.zeros((1, 1, self.num_tokens))
            target_seq[0, 0, self.vectorizer.character_index[self.vectorizer.SOS]] = 1.

            pred = self.inference_model.predict([image] + [target_seq])

            text = ''
            samples = np.argmax(pred, axis=2)[0]
            for sample_index in samples:
                sample = self.vectorizer.character_reverse_index[sample_index]
                if sample == self.vectorizer.EOS or sample == self.vectorizer.PAD or len(text) > self.max_output_txt_size:
                    break
                text += sample
            texts.append(text)
        return texts


class Encoder:
    def __init__(self, units):
        layers = [Conv2D(64, (3, 3), padding='same', activation='relu'),
                  MaxPool2D(strides=(2, 2), padding='valid'),
                  BatchNormalization(),
                  Conv2D(128, (3, 3), padding='same', activation='relu'),
                  MaxPool2D(strides=(2, 2), padding='valid'),
                  BatchNormalization(),
                  Conv2D(256, (3, 3), padding='same', activation='relu'),
                  MaxPool2D(strides=(2, 2), padding='valid'),
                  BatchNormalization(),
                  Conv2D(256, (3, 3), padding='valid', activation='relu'),
                  MaxPool2D(strides=(2, 2), padding='valid'),
                  BatchNormalization()]

        self.cnn = Sequential(layers)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)

    def __call__(self, encoder_input):
        out = self.cnn(encoder_input)
        out = tf.squeeze(out, axis=1)
        return self.lstm(out)


class Attention:
    def __init__(self, units: int):
        # https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

        self.projection_q = Dense(units)

    def __call__(self, decoder_input, encoder_states):
        Q = self.projection_q(decoder_input)
        K = encoder_states
        V = encoder_states

        logits = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1]))
        attention_weights = tf.nn.softmax(logits)
        context_vectors = tf.matmul(attention_weights, V)
        return [context_vectors, attention_weights]


class Decoder:
    def __init__(self, units):
        self.lstm = LSTM(units, return_sequences=True, return_state=True)

    def __call__(self, decoder_input, initial_state):
        return self.lstm(decoder_input, initial_state)


class Output:
    def __init__(self, vocab_size):
        self.dense = Dense(vocab_size, activation='softmax')

    def __call__(self, decoder_output):
        return self.dense(decoder_output)

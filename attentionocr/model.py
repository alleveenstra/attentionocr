import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Bidirectional, Concatenate, Conv2D, Activation, RepeatVector, Permute, LSTM, Multiply, BatchNormalization, Flatten, Dense, MaxPooling2D, TimeDistributed, Dot, Softmax, Lambda
from tensorflow.python.keras.models import Model
import tensorflow as tf

import numpy as np
# from tensorflow.python.keras.layers.advanced_activations import Softmax

from .vectorizer import VectorizerOCR


class KerasAttentionOCR:

    def __init__(self, vectorizer: VectorizerOCR, latent_dim=256, image_height=32, image_width=144, bidirectional_encoder=False):
        self.vectorizer = vectorizer
        self.num_tokens = len(vectorizer.tokens())
        self.max_input_txt_size = self.vectorizer.max_txt_length
        self.max_output_txt_size = self.vectorizer.max_txt_length + 2
        self.latent_dim = latent_dim
        self.image_height = image_height
        self.image_width = image_width
        self.bidirectional_encoder = bidirectional_encoder
        self._build_model()

    def _build_model(self):
        encoder_inputs = Input(shape=(self.image_height, self.image_width, 1), name="input_encoder")
        decoder_inputs = Input(shape=(None, self.num_tokens,), name="input_decoder")

        encoder_hidden_states, encoder_states = self._build_encoder(encoder_inputs)

        attention = Attention(self.latent_dim)

        context_vector, _ = attention(decoder_inputs, encoder_hidden_states)

        decoder_dense, decoder_lstm, decoder_outputs = self._build_decoder(decoder_inputs, context_vector, encoder_states)

        # training_model: encoder_inputs, decoder_inputs => decoder_outputs
        self.training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.training_model.summary()

        # inference_encoder: encoder_inputs => encoder_states
        self.inference_encoder = Model(encoder_inputs, encoder_states)
        self.inference_decoder = self._build_inference_decoder(encoder_inputs, decoder_dense, decoder_inputs, decoder_lstm, attention, encoder_states, encoder_hidden_states)

    def _build_inference_decoder(self, encoder_inputs, decoder_dense, decoder_inputs, decoder_lstm, attention, encoder_states, encoder_hidden_states):
        predictions = []
        prediction = decoder_inputs
        state_h, state_c = encoder_states
        for i in range(self.max_input_txt_size):
            context_vectors, _ = attention(prediction, encoder_hidden_states)
            x = Concatenate(axis=2)([prediction, context_vectors])
            decoder_output, state_h, state_c = decoder_lstm(x, initial_state=[state_h, state_c])
            prediction = decoder_dense(decoder_output)
            predictions.append(prediction)
        probabilities = Concatenate(axis=1)(predictions)
        return tf.keras.Model([encoder_inputs, decoder_inputs], probabilities)

    def _build_encoder(self, input_image_tensor):
        encoder = Conv2D(44, (32, 1), padding='valid', activation='relu')(input_image_tensor)
        assert encoder.shape[1] == 1
        encoder = K.squeeze(encoder, axis=1)
        activations, state_h, state_c = LSTM(self.latent_dim, return_sequences=True, return_state=True)(encoder)
        return activations, [state_h, state_c]

    def _build_decoder(self, decoder_inputs, context_vector, encoder_state):

        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)

        x = Concatenate(axis=2)([decoder_inputs, context_vector])
        activations, _, _ = decoder_lstm(x, initial_state=encoder_state)

        decoder_dense = Dense(self.num_tokens, activation='softmax')
        decoder_outputs = decoder_dense(activations)

        return decoder_dense, decoder_lstm, decoder_outputs

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

            pred = self.inference_decoder.predict([image] + [target_seq])

            text = ''
            samples = np.argmax(pred, axis=2)[0]
            for sample_index in samples:
                sample = self.vectorizer.character_reverse_index[sample_index]
                if sample == self.vectorizer.EOS or sample == self.vectorizer.PAD or len(text) > self.max_output_txt_size:
                    break
                text += sample
            texts.append(text)
        return texts


class Attention:
    def __init__(self, units: int):
        # https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
        self.projection = Dense(units, activation="sigmoid")

    def __call__(self, decoder_inputs, encoder_hidden_states):
        query = self.projection(decoder_inputs)
        key = encoder_hidden_states
        value = encoder_hidden_states
        key = Permute([2, 1])(key)
        attention = Dot(axes=(2, 1))([query, key])
        attention_weights = Softmax(axis=1)(attention)
        context_vector = Dot(axes=(2, 1))([attention_weights, value])
        return context_vector, attention_weights

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
        decoder_inputs = Input(shape=(self.max_input_txt_size, self.num_tokens,), name="input_decoder")

        encoder_hidden_states, encoder_states = self._build_encoder(encoder_inputs)
        decoder_dense, decoder_lstm, decoder_outputs = self._build_decoder(decoder_inputs, encoder_hidden_states, encoder_states)

        # training_model: encoder_inputs, decoder_inputs => decoder_outputs
        self.training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.training_model.summary()

        # inference_encoder: encoder_inputs => encoder_states
        print(encoder_states)
        self.inference_encoder = Model(encoder_inputs, encoder_states)
        self.inference_decoder = self._build_inference_decoder(decoder_dense, decoder_inputs, decoder_lstm)

    def _build_inference_decoder(self, decoder_dense, decoder_inputs, decoder_lstm):
        hidden_state_input = Input(shape=(self.latent_dim,), name="decoder_hidden_state")
        cell_state_input = Input(shape=(self.latent_dim,), name="decoder_cell_state")
        decoder_states_inputs = [hidden_state_input, cell_state_input]
        decoder_outputs, hidden_state, cell_state = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [hidden_state, cell_state]
        decoder_outputs = decoder_dense(decoder_outputs)
        inference_decoder = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return inference_decoder

    def _build_encoder(self, input_image_tensor):
        encoder = Conv2D(44, (32, 1), padding='valid', activation='relu')(input_image_tensor)
        assert encoder.shape[1] == 1
        encoder = K.squeeze(encoder, axis=1)
        activations, state_h, state_c = LSTM(self.latent_dim, return_sequences=True, return_state=True)(encoder)
        print(activations)
        print(state_h.shape, state_c.shape)

        return activations, [state_h, state_c]

    def _build_decoder(self, decoder_inputs, encoder_hidden_states, encoder_state):


        print('encoder_state[0]', encoder_state[0].shape)

        # attn = Attention()([encoder_hidden_states, aap])  # maybe other way around

        # https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
        Q = Dense(self.latent_dim)(decoder_inputs)
        Key = encoder_hidden_states
        V = encoder_hidden_states
        KT = Permute([2, 1])(Key)
        QKT = Dot(axes=(2, 1))([Q, KT])
        AQKT = Softmax(axis=1)(QKT)
        AQKTV = Dot(axes=(2, 1))([AQKT, V])

        # # attention
        # attention = TimeDistributed(Dense(1, activation='tanh'))(encoder_hidden_states)
        # attention = K.squeeze(attention, axis=2)
        # attention = Activation('softmax')(attention)
        # attention = RepeatVector(activations.shape[-1])(attention)
        # attention = Permute([2, 1])(attention)
        #
        # applied_attention = Multiply()([activations, attention])
        # # applied_attention = K.sum(applied_attention, axis=2)

        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)

        activations = []
        cell_state = tf.convert_to_tensor(encoder_state[1])
        for i in range(self.max_input_txt_size):
            state_h = Lambda(lambda x: x)(AQKTV[:, i, :])
            inpt = tf.expand_dims(decoder_inputs[:, i, :], axis=1)
            act, _, cell_state = decoder_lstm(inpt, initial_state=[state_h, cell_state])
            print('act', act.shape)
            activations.append(act)
        activations = Concatenate(axis=1)(activations)

        print('activations.shape', activations.shape)

        decoder_dense = Dense(self.num_tokens, activation='softmax')
        decoder_outputs = decoder_dense(activations)

        print('#', decoder_outputs.shape)

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
            states_value = self.inference_encoder.predict(image)

            target_seq = np.zeros((1, 1, self.num_tokens))
            target_seq[0, 0, self.vectorizer.character_index[self.vectorizer.SOS]] = 1.

            text = ""
            while True:
                output_tokens, state_h, state_c = self.inference_decoder.predict([target_seq] + states_value)
                states_value = [state_h, state_c]  # loop the state back for the next sample

                # greedy search
                sample_index = np.argmax(output_tokens[0, -1, :])
                sample = self.vectorizer.character_reverse_index[sample_index]
                if sample == self.vectorizer.EOS or sample == self.vectorizer.PAD or len(text) > self.max_output_txt_size:
                    break
                text += sample

                target_seq = np.zeros((1, 1, self.num_tokens))
                target_seq[0, 0, sample_index] = 1.

            texts.append(text)
        return texts

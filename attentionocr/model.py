import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Bidirectional, Concatenate, Conv2D, Activation, RepeatVector, Permute, LSTM, Multiply, BatchNormalization, Flatten, Dense, MaxPooling2D, TimeDistributed
from tensorflow.python.keras.models import Model

import numpy as np

from .vectorizer import VectorizerOCR


class KerasAttentionOCR:

    def __init__(self, vectorizer: VectorizerOCR):
        self.vectorizer = vectorizer
        self.num_tokens = len(vectorizer.tokens())
        self.max_input_txt_size = self.vectorizer.max_txt_length
        self.max_output_txt_size = self.vectorizer.max_txt_length + 2
        self.latent_dim = 256
        self.image_height = 32
        self.image_width = 144
        self._build_model()

    def _build_model(self):
        encoder_inputs = Input(shape=(self.image_height, self.image_width, 1), name="input_encoder")
        decoder_inputs = Input(shape=(None, self.num_tokens,), name="input_decoder")

        encoder_states = self._build_encoder(encoder_inputs)
        decoder_dense, decoder_lstm, decoder_outputs = self._build_decoder(decoder_inputs, encoder_states)

        # training_model: encoder_inputs, decoder_inputs => decoder_outputs
        self.training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.training_model.summary()

        # inference_encoder: encoder_inputs => encoder_states
        self.inference_encoder = Model(encoder_inputs, encoder_states)
        self.inference_decoder = self._build_inference_decoder(decoder_dense, decoder_inputs, decoder_lstm)

    def _build_inference_decoder(self, decoder_dense, decoder_inputs, decoder_lstm):
        input_forward_h = Input(shape=(self.latent_dim,))
        input_forward_c = Input(shape=(self.latent_dim,))
        input_backward_h = Input(shape=(self.latent_dim,))
        input_backward_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [input_forward_h, input_forward_c, input_backward_h, input_backward_c]
        decoder_outputs, forward_h, forward_c, backward_h, backward_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [forward_h, forward_c, backward_h, backward_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        inference_decoder = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return inference_decoder

    def _build_encoder(self, input_image_tensor):
        conv = Conv2D(32, (3, 3), padding='same', activation='relu')(input_image_tensor)
        conv = MaxPooling2D((2, 2))(conv)
        conv = BatchNormalization(axis=3)(conv)
        conv = Conv2D(64, (3, 3), padding='same', activation='relu')(conv)
        conv = MaxPooling2D((2, 2))(conv)
        conv = BatchNormalization(axis=3)(conv)
        conv = Conv2D(128, (3, 3), padding='same', activation='relu')(conv)
        conv = MaxPooling2D((2, 2))(conv)
        conv = BatchNormalization(axis=3)(conv)
        conv = TimeDistributed(Flatten())(conv)
        encoder = Bidirectional(LSTM(self.latent_dim, return_state=True))
        _, forward_h, forward_c, backward_h, backward_c = encoder(conv)
        encoder_states = [forward_h, forward_c, backward_h, backward_c]
        return encoder_states

    def _build_decoder(self, decoder_inputs, encoder_states):
        # state is used during inference, but ignored for now
        decoder_lstm = Bidirectional(LSTM(self.latent_dim, return_sequences=True, return_state=True))
        activations, _, _, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        # attention
        attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
        attention = K.squeeze(attention, axis=2)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(activations.shape[-1])(attention)
        attention = Permute([2, 1])(attention)

        applied_attention = Multiply()([activations, attention])
        # applied_attention = K.sum(applied_attention, axis=2)
        decoder_dense = Dense(self.num_tokens, activation='softmax')
        decoder_outputs = decoder_dense(applied_attention)
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
                output_tokens, forward_h, forward_c, backward_h, backward_c = self.inference_decoder.predict([target_seq] + states_value)
                states_value = [forward_h, forward_c, backward_h, backward_c]  # loop the state back for the next sample

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

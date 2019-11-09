import tensorflow.keras.backend as K
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import MaxPool2D, Bidirectional, Concatenate, Conv2D, Activation, RepeatVector, Permute, LSTM, Multiply, BatchNormalization, Flatten, Dense, MaxPooling2D, TimeDistributed, Dot, Softmax, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf

import numpy as np
from tqdm import tqdm

from attentionocr import metrics
from .vectorizer import VectorizerOCR, VectorizedBatchGenerator


class AttentionOCR:

    def __init__(self, vectorizer: VectorizerOCR, units=256):
        self.vectorizer = vectorizer
        self.num_tokens = len(vectorizer.tokens())
        self.max_input_txt_size = self.vectorizer.max_txt_length
        self.max_output_txt_size = self.vectorizer.max_txt_length + 2
        self.image_height = 32

        self.units = units

        # Build the model.
        self.encoder_input = Input(shape=(self.image_height, None, 1), name="encoder_input")
        self.decoder_input = Input(shape=(None, self.num_tokens), name="decoder_input")
        
        self.inference_encoder_output = Input(shape=(None, self.units), name="inference_encoder_output")
        self.inference_hidden_state = Input(shape=(self.units,), name="inference_hidden_state")
        self.inference_cell_state = Input(shape=(self.units,), name="inference_cell_state")

        self.encoder = Encoder(self.units)
        self.attention = Attention(self.units)
        self.decoder = Decoder(self.units)
        self.output = Output(self.num_tokens)

        self.training_model = self.build_training_model()
        self.inference_encoder = self.build_inference_encoder_model()
        self.inference_decoder = self.build_inference_decoder_model()

    def build_training_model(self) -> tf.keras.Model:
        encoder_output, hidden_state, cell_state = self.encoder(self.encoder_input)
        initial_state = [hidden_state, cell_state]
        context_vectors, _ = self.attention(self.decoder_input, encoder_output)
        x = tf.concat([self.decoder_input, context_vectors], axis=2)
        decoder_output, _, _ = self.decoder(x, initial_state=initial_state)
        scores = self.output(decoder_output)
        return tf.keras.Model([self.encoder_input, self.decoder_input], scores)

    def build_inference_encoder_model(self) -> tf.keras.Model:
        encoder_output, state_h, state_c = self.encoder(self.encoder_input)
        return Model(self.encoder_input, [encoder_output, state_h, state_c])

    def build_inference_decoder_model(self) -> tf.keras.Model:
        initial_state = [self.inference_hidden_state, self.inference_cell_state]
        context_vectors, _ = self.attention(self.decoder_input, self.inference_encoder_output)
        x = tf.concat([self.decoder_input, context_vectors], axis=2)
        decoder_output, hidden_state, cell_state = self.decoder(x, initial_state=initial_state)
        scores = self.output(decoder_output)
        return tf.keras.Model([self.inference_encoder_output, self.inference_hidden_state, self.inference_cell_state, self.decoder_input], [scores, hidden_state, cell_state])

    def fit_generator(self, generator, steps_per_epoch: int = 1, epochs: int = 1, validation_data=None):
        optimizer = tf.optimizers.RMSprop()
        loss_function = tf.losses.categorical_crossentropy
        K.set_learning_phase(1)
        for epoch in range(epochs):
            pbar = tqdm(range(steps_per_epoch))
            for _ in pbar:
                x, y_true = next(generator)
                with tf.GradientTape() as tape:
                    predictions = self.training_model(x)
                    loss = tf.reduce_mean(loss_function(y_true, predictions))
                variables = self.training_model.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
            if validation_data is not None:
                x, y_true = next(validation_data)
                y_pred = self.training_model(x)
                accuracy = metrics.masked_accuracy(y_true, y_pred)
                pbar.set_description("Test accuracy %.04f" % accuracy)

    def fit(self, images: list, texts: list, epochs: int = 10, batch_size: int = None, validation_split=0.):
        self.training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.training_model.summary()
        if batch_size is None:
            batch_size = len(images)
        X, y = self.vectorizer.vectorize(images, texts)
        K.set_learning_phase(1)
        self.training_model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def predict(self, images) -> list:
        K.set_learning_phase(0)
        texts = []
        for image in images:
            # feed the input, retrieve encoder state vectors
            image = np.expand_dims(image, axis=0)

            # encoder_input -> [encoder_output, state_h, state_c]
            encoder_output, state_h, state_c = self.inference_encoder.predict(image)

            decoder_input = np.zeros((1, 1, self.num_tokens))
            decoder_input[0, 0, self.vectorizer.character_index[self.vectorizer.SOS]] = 1.

            text = ""
            while True:
                # [self.encoder_output, self.hidden_state_input, self.cell_state_input, self.decoder_input] -> [scores, hidden_state, cell_state]
                decoder_output, state_h, state_c = self.inference_decoder.predict([encoder_output, state_h, state_c, decoder_input])

                # greedy search
                sample_index = np.argmax(decoder_output[0, -1, :])
                sample = self.vectorizer.character_reverse_index[sample_index]
                if sample == self.vectorizer.EOS or sample == self.vectorizer.PAD or len(text) > self.max_output_txt_size:
                    break
                text += sample

                # loop the decoder output into the decoder input
                decoder_input = decoder_output

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
        self.query_projection = Dense(units)

    def __call__(self, decoder_input, encoder_output):
        query = self.query_projection(decoder_input)
        key = encoder_output
        value = encoder_output
        logits = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))
        attention_weights = tf.nn.softmax(logits)
        context_vectors = tf.matmul(attention_weights, value)
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

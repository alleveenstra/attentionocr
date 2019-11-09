from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import MaxPool2D, Conv2D, LSTM, BatchNormalization, Dense
from tensorflow.keras.models import Model
from tqdm import tqdm

from attentionocr import metrics
from .vectorizer import VectorizerOCR


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
        self.inference_model = self.build_inference_model()

    def build_inference_model(self) -> tf.keras.Model:
        predictions = []
        prediction = self.decoder_input
        encoder_output, hidden_state, cell_state = self.encoder(self.encoder_input)
        for i in range(self.max_output_txt_size):
            context_vectors, attention_weights = self.attention(prediction, encoder_output)
            x = tf.concat([prediction, context_vectors], axis=2)
            decoder_output, hidden_state, cell_state = self.decoder(x, [hidden_state, cell_state])
            prediction = self.output(decoder_output)
            predictions.append(prediction)
        logits = tf.concat(predictions, axis=1)
        return tf.keras.Model([self.encoder_input, self.decoder_input], logits)

    def build_training_model(self) -> tf.keras.Model:
        encoder_output, hidden_state, cell_state = self.encoder(self.encoder_input)
        initial_state = [hidden_state, cell_state]
        context_vectors, _ = self.attention(self.decoder_input, encoder_output)
        x = tf.concat([self.decoder_input, context_vectors], axis=2)
        decoder_output, _, _ = self.decoder(x, initial_state=initial_state)
        logits = self.output(decoder_output)
        return tf.keras.Model([self.encoder_input, self.decoder_input], logits)

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

    def fit_generator(self, generator, steps_per_epoch: int = 1, epochs: int = 1, validation_data=None, validate_every_steps: int = 10) -> None:
        optimizer = tf.optimizers.RMSprop()
        loss_function = tf.losses.categorical_crossentropy
        K.set_learning_phase(1)
        accuracy = 0
        for epoch in range(epochs):
            print("Epoch %d / %d..." % (epoch, epochs))
            pbar = tqdm(range(steps_per_epoch))
            for step in pbar:
                x, y_true = next(generator)
                with tf.GradientTape() as tape:
                    predictions = self.training_model(x)
                    loss = tf.reduce_mean(loss_function(y_true, predictions))
                variables = self.training_model.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                if step % validate_every_steps == 0 and validation_data is not None:
                    x, y_true = next(validation_data)
                    y_pred = self.inference_model(x)
                    accuracy = metrics.masked_accuracy(y_true, y_pred)
                pbar.set_postfix({"test_accuracy": "%.04f" % accuracy, "loss": loss.numpy()})

    def fit(self, images: list, texts: list, epochs: int = 10, batch_size: int = None, validation_split=0.) -> None:
        self.training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.training_model.summary()
        if batch_size is None:
            batch_size = len(images)
        X, y = self.vectorizer.vectorize(images, texts)
        K.set_learning_phase(1)
        self.training_model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def save(self, filepath) -> None:
        self.training_model.save_weights(filepath=filepath)

    def load(self, filepath) -> None:
        self.training_model.load_weights(filepath=filepath)

    def predict(self, images) -> List[str]:
        K.set_learning_phase(0)
        texts = []
        for image in images:
            # feed the input, retrieve encoder state vectors
            image = np.expand_dims(image, axis=0)

            decoder_input = np.zeros((1, 1, self.num_tokens))
            decoder_input[0, 0, self.vectorizer.character_index[self.vectorizer.SOS]] = 1.

            text = ""
            decoder_output = self.inference_model.predict([image, decoder_input])

            # greedy search
            greedy_matches = np.argmax(decoder_output, axis=-1)[0]
            for sample_index in greedy_matches:
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

    def __call__(self, encoder_input) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        out = self.cnn(encoder_input)
        out = tf.squeeze(out, axis=1)
        return self.lstm(out)


class Attention:
    def __init__(self, units: int):
        # https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
        self.query_projection = Dense(units)

    def __call__(self, decoder_input, encoder_output) -> Tuple[tf.Tensor, tf.Tensor]:
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

    def __call__(self, decoder_input, initial_state) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.lstm(decoder_input, initial_state)


class Output:
    def __init__(self, vocab_size):
        self.dense = Dense(vocab_size, activation='softmax')

    def __call__(self, decoder_output) -> tf.Tensor:
        return self.dense(decoder_output)

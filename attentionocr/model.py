import datetime
import os
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import MaxPool2D, Conv2D, LSTM, BatchNormalization, Dense
from tqdm import tqdm

from attentionocr import metrics, Vocabulary


class AttentionOCR:

    def __init__(self, vocabulary: Vocabulary, max_txt_length: int = 42, units: int = 256):
        self._vocabulary = vocabulary
        self._max_txt_length = max_txt_length
        self._image_height = 32
        self._units = units

        log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)

        # Build the model.
        self._encoder_input = Input(shape=(self._image_height, None, 1), name="encoder_input")
        self._decoder_input = Input(shape=(None, len(self._vocabulary)), name="decoder_input")

        self._encoder = Encoder(self._units)
        self._attention = Attention(self._units)
        self._decoder = Decoder(self._units)
        self._output = DecoderOutput(len(self._vocabulary))

        self._training_model = self.build_training_model()
        self._inference_model = self.build_inference_model()
        self._visualisation_model = self.build_inference_model(include_attention=True)

    def build_training_model(self) -> tf.keras.Model:
        encoder_output, hidden_state, cell_state = self._encoder(self._encoder_input)
        initial_state = [hidden_state, cell_state]
        context_vectors, _ = self._attention(self._decoder_input, encoder_output)
        x = tf.concat([self._decoder_input, context_vectors], axis=2)
        decoder_output, _, _ = self._decoder(x, initial_state=initial_state)
        logits = self._output(decoder_output)
        return tf.keras.Model([self._encoder_input, self._decoder_input], logits)

    def build_inference_model(self, include_attention: bool = False) -> tf.keras.Model:
        predictions = []
        attentions = []
        prediction = self._decoder_input
        encoder_output, hidden_state, cell_state = self._encoder(self._encoder_input)
        for i in range(self._max_txt_length):
            context_vectors, attention = self._attention(prediction, encoder_output)
            attentions.append(attention)
            x = tf.concat([prediction, context_vectors], axis=2)
            decoder_output, hidden_state, cell_state = self._decoder(x, [hidden_state, cell_state])
            prediction = self._output(decoder_output)
            predictions.append(prediction)
        predictions = tf.concat(predictions, axis=1)
        attentions = tf.concat(attentions, axis=1)
        output = [predictions, attentions] if include_attention else predictions
        return tf.keras.Model([self._encoder_input, self._decoder_input], output)

    def fit_generator(self, generator, steps_per_epoch: int = 1, epochs: int = 1, validation_data=None, validate_every_steps: int = 10) -> None:
        optimizer = tf.optimizers.RMSprop()
        loss_function = tf.losses.CategoricalCrossentropy()
        K.set_learning_phase(1)
        for epoch in range(epochs):
            tensorboard_loss = tf.keras.metrics.Mean()
            tensorboard_accuracy = tf.keras.metrics.Mean()
            pbar = tqdm(range(steps_per_epoch))
            pbar.set_description("Epoch %03d / %03d " % (epoch, epochs))
            for step in pbar:
                x, y_true = next(generator)
                with tf.GradientTape() as tape:
                    predictions = self._training_model(x)
                    loss = loss_function(y_true, predictions)
                    tensorboard_loss(loss)
                variables = self._training_model.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                if step % validate_every_steps == 0 and validation_data is not None:
                    x, y_true = next(validation_data)
                    y_pred = self._inference_model(x)
                    accuracy = metrics.masked_accuracy(y_true, y_pred)
                    tensorboard_accuracy(accuracy)
                    pbar.set_postfix({"accuracy": "%.4f" % accuracy, "loss": "%.4f" % loss.numpy()})
                with self.summary_writer.as_default():
                    tf.summary.scalar('epoch_loss_avg', tensorboard_loss.result(), step=optimizer.iterations)
                    tf.summary.scalar('epoch_accuracy', tensorboard_accuracy.result(), step=optimizer.iterations)

    def save(self, filepath) -> None:
        self._training_model.save_weights(filepath=filepath)

    def load(self, filepath) -> None:
        self._training_model.load_weights(filepath=filepath)

    def predict(self, images) -> List[str]:
        K.set_learning_phase(0)
        texts = []
        for image in images:
            image = np.expand_dims(image, axis=0)

            decoder_input = np.zeros((1, 1, len(self._vocabulary)))
            decoder_input[0, :, :] = self._vocabulary.one_hot_encode('', 1, sos=True, eos=False)

            y_pred = self._inference_model.predict([image, decoder_input])
            texts.append(self._vocabulary.one_hot_decode(y_pred, self._max_txt_length))
        return texts

    def visualise(self, images) -> List[str]:
        K.set_learning_phase(0)
        texts = []
        for image in images[:1]:
            input_image = np.expand_dims(image, axis=0)

            decoder_input = np.zeros((1, 1, len(self._vocabulary)))
            decoder_input[0, :, :] = self._vocabulary.one_hot_encode('', 1, sos=True, eos=False)

            y_pred, attention = self._visualisation_model.predict([input_image, decoder_input])
            text = self._vocabulary.one_hot_decode(y_pred, self._max_txt_length)

            step_size = float(image.shape[1]) / attention.shape[-1]
            for index, char_idx in enumerate(np.argmax(y_pred, axis=-1)[0]):
                if char_idx < 3:  # TODO magic value for PAD, EOS, SOS
                    break
                heatmap = np.zeros(image.shape)
                for location, strength in enumerate(attention[0, index, :]):
                    heatmap[:, int(location * step_size) : int((location + 1) * step_size)] = strength * 255.0
                filtered_image = (image + 1.0) * 127.5 * 0.4 + heatmap * 0.6
                cv2.imwrite('out/%s-%d-%s.png' % (text, index, text[index]), filtered_image)

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
                  MaxPool2D(strides=(2, 1), padding='valid'),
                  BatchNormalization(),
                  Conv2D(256, (3, 3), padding='valid', activation='relu'),
                  MaxPool2D(strides=(2, 1), padding='valid'),
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


class DecoderOutput:
    def __init__(self, vocab_size):
        self.dense = Dense(vocab_size, activation='softmax')

    def __call__(self, decoder_output) -> tf.Tensor:
        return self.dense(decoder_output)

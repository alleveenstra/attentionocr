import tensorflow as tf
from attentionocr import Vectorizer, AttentionOCR, synthetic_data_generator, Vocabulary, CSVDataSource

if __name__ == "__main__":
    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_width=320, max_txt_length=42)
    model = AttentionOCR(vocabulary=voc, max_txt_length=42)

    # train_data = synthetic_data_generator(vec, epoch_size=8, augment=False, is_training=True)
    # validation_data = synthetic_data_generator(vec, epoch_size=8, augment=False)
    train_data = CSVDataSource(vec, 'data/', 'train.txt', is_training=True)
    validation_data = CSVDataSource(vec, 'data/', 'validation.txt')

    train_gen = tf.data.Dataset.from_generator(train_data, output_types=(tf.float32, tf.float32, tf.float32))
    validation_gen = tf.data.Dataset.from_generator(validation_data, output_types=(tf.float32, tf.float32, tf.float32))

    # model.load('model.h5')
    model.fit_generator(train_gen, epochs=1, batch_size=8, validation_data=validation_gen, validate_every_steps=10)
    model.save('model.h5')

    validation_data = synthetic_data_generator(vec, epoch_size=2, augment=False, is_training=True)
    for image, decoder_input, decoder_output in validation_data():
        txt = voc.one_hot_decode(decoder_output, 42)
        pred = model.predict([image])[0]
        model.visualise([image])
        print(txt, "prediction: ", pred)

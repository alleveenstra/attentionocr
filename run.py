import tensorflow as tf
from attentionocr import Vectorizer, AttentionOCR, synthetic_data_generator, Vocabulary, FlatDirectoryDataSource

if __name__ == "__main__":
    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_width=320, max_txt_length=42)
    model = AttentionOCR(vocabulary=voc, max_txt_length=42, focus_attention=False)
    train_data = synthetic_data_generator(vec, epoch_size=16, augment=True, is_training=True)
    test_data = synthetic_data_generator(vec, epoch_size=16)

    train_gen = tf.data.Dataset.from_generator(train_data, output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
    test_gen = tf.data.Dataset.from_generator(test_data, output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

    model.fit_generator(train_gen, epochs=1, batch_size=8, validation_data=test_gen, validate_every_steps=1)

    model.save('model.h5')

    for image, decoder_input, decoder_output, position, focus in test_data():

        # import numpy as np
        # image = np.squeeze(image, axis=-1)
        # import matplotlib.pyplot as plt
        # print(text)
        # imgplot = plt.imshow(image)
        # plt.show()
        # os.exit(-1)

        pred = model.predict([image])[0]
        model.visualise([image])
        print("prediction: ", pred)

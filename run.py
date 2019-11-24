import tensorflow as tf
from attentionocr import Vectorizer, AttentionOCR, CSVDataSource, Vocabulary, FlatDirectoryDataSource

if __name__ == "__main__":
    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_width=320, max_txt_length=42)
    model = AttentionOCR(vocabulary=voc, max_txt_length=42, focus_attention=True)
    train_data = FlatDirectoryDataSource(vec, 'scripts/train/*.jpg', is_training=True)
    test_data = FlatDirectoryDataSource(vec, 'scripts/test/*.jpg')

    train_gen = tf.data.Dataset.from_generator(train_data, output_types=(tf.float32, tf.float32, tf.float32, tf.float32))
    test_gen = tf.data.Dataset.from_generator(test_data, output_types=(tf.float32, tf.float32, tf.float32, tf.float32))

    model.fit_generator(train_gen, epochs=10, validation_data=test_gen, validate_every_steps=10)

    model.save('model.h5')

    for i in range(1):
        filename, text = next(test_data)
        image = vec._image_util.load(filename)

        # import numpy as np
        # image = np.squeeze(image, axis=-1)
        # import matplotlib.pyplot as plt
        # print(text)
        # imgplot = plt.imshow(image)
        # plt.show()
        # os.exit(-1)

        pred = model.predict([image])[0]
        model.visualise([image])
        print('Input:', text, " prediction: ", pred)

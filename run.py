import string

from attentionocr import VectorizerOCR, AttentionOCR, FlatDirectoryIterator, Vocabulary
from attentionocr.vectorizer import VectorizedBatchGenerator

if __name__ == "__main__":
    voc = Vocabulary(list(string.ascii_lowercase) + list(string.digits))
    vec = VectorizerOCR(vocabulary=voc, image_width=320, max_txt_length=42)
    model = AttentionOCR(vectorizer=vec, vocabulary=voc)
    train_data = list(FlatDirectoryIterator('train/*.jpg'))
    test_data = list(FlatDirectoryIterator('test/*.jpg'))

    # generator = VectorizedBatchGenerator(vectorizer=vec)
    # train_bgen = generator.flow_from_dataset(train_data)
    # test_bgen = generator.flow_from_dataset(test_data, is_training=False)
    # model.fit_generator(train_bgen, epochs=1, steps_per_epoch=1, validation_data=test_bgen)

    model.load('model.h5')

    for i in range(2):
        filename, text = test_data[i]
        image = vec.load_image(filename)
        pred = model.predict([image])[0]
        model.visualise([image])
        print('Input:', text, " prediction: ", pred)

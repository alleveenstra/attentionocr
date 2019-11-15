import string

from attentionocr import Vectorizer, AttentionOCR, CSVDataSource, Vocabulary, BatchGenerator


if __name__ == "__main__":
    voc = Vocabulary(list(string.ascii_lowercase) + list(string.digits) + [' ', '-', '.', ':', '?', '!', '<', '>', '#', '@', '(', ')', '$', '%', '&'])
    vec = Vectorizer(vocabulary=voc, image_width=320, max_txt_length=42)
    model = AttentionOCR(vocabulary=voc, max_txt_length=42)
    train_data = CSVDataSource('/home/alle/CRNN_6/Train/', 'sample.txt')
    test_data = CSVDataSource('/home/alle/CRNN_6/Validation/', 'sample.txt')

    model.load('model.h5')

    generator = BatchGenerator(vectorizer=vec, batch_size=512)
    train_bgen = generator.flow_from_datasource(train_data)
    test_bgen = generator.flow_from_datasource(test_data, is_training=False)
    model.fit_generator(train_bgen, epochs=4000, steps_per_epoch=100, validation_data=test_bgen)

    model.save('model2.h5')

    for i in range(1):
        filename, text = test_data[i]
        image = vec._image_util.load(filename)
        pred = model.predict([image])[0]
        model.visualise([image])
        print('Input:', text, " prediction: ", pred)

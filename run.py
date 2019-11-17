import string

from attentionocr import Vectorizer, AttentionOCR, CSVDataSource, Vocabulary, BatchGenerator, FlatDirectoryDataSource

if __name__ == "__main__":
    voc = Vocabulary(list(string.ascii_lowercase) + list(string.digits) + [' ', '-', '.', ':', '?', '!', '<', '>', '#', '@', '(', ')', '$', '%', '&'])
    vec = Vectorizer(vocabulary=voc, image_width=320, max_txt_length=42)
    model = AttentionOCR(vocabulary=voc, max_txt_length=42, focus_attention=True)
    train_data = FlatDirectoryDataSource('scripts/train/*.jpg')
    test_data = FlatDirectoryDataSource('scripts/test/*.jpg')

    generator = BatchGenerator(vectorizer=vec, batch_size=64)
    train_bgen = generator.flow_from_datasource(train_data)
    test_bgen = generator.flow_from_datasource(test_data, is_training=False)
    model.fit_generator(train_bgen, epochs=10, steps_per_epoch=10, validation_data=test_bgen)

    # model.load('model.h5')
    model.save('model.h5')

    for i in range(1):
        filename, text = next(test_data)
        image = vec._image_util.load(filename)
        pred = model.predict([image])[0]
        model.visualise([image])
        print('Input:', text, " prediction: ", pred)

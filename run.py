import string

from attentionocr import VectorizerOCR, AttentionOCR, FlatDirectoryIterator
from attentionocr.vectorizer import VectorizedBatchGenerator

if __name__ == "__main__":
    vec = VectorizerOCR(vocabulary=list(string.ascii_lowercase) + list(string.digits))
    model = AttentionOCR(vectorizer=vec)
    train_data = list(FlatDirectoryIterator('train/*.jpg'))
    test_data = list(FlatDirectoryIterator('test/*.jpg'))

    # images, texts = zip(*train_data)
    # model.fit(images=images, texts=texts, epochs=1, batch_size=64)

    batch_generator = VectorizedBatchGenerator(vectorizer=vec).flow_from_dataset(train_data)
    model.fit_generator(batch_generator, epochs=1)

    for i in range(10):
        filename, text = test_data[i]
        image = vec.load_image(filename)
        pred = model.predict([image])[0]
        print('Input:', text, " prediction: ", pred)

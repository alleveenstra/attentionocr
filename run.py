import string

from attentionocr import VectorizerOCR, KerasAttentionOCR, FlatDirectoryIterator

if __name__ == "__main__":
    vec = VectorizerOCR(vocabulary=list(string.ascii_uppercase) + list(string.digits))
    model = KerasAttentionOCR(vectorizer=vec)
    train_data = list(FlatDirectoryIterator('train/*.jpg'))
    test_data = list(FlatDirectoryIterator('test/*.jpg'))

    images, texts = zip(*train_data)
    model.fit(images, texts, epochs=1, batch_size=64, validation_split=0.2)

    for i in range(10):
        filename, text = test_data[i]
        image = vec.load_image(filename)
        pred = model.predict([image])[0]
        print('Input:', text, " prediction: ", pred)

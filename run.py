import argparse

import tensorflow as tf
from attentionocr import Vectorizer, AttentionOCR, synthetic_data_generator, Vocabulary, CSVDataSource

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--epoch_size', type=int, default=8, required=False)
    parser.add_argument('--image_width', type=int, default=320, required=False)
    parser.add_argument('--max_txt_length', type=int, default=42, required=False)
    parser.add_argument('--batch_size', type=int, default=8, required=False)
    parser.add_argument('--validate_every_steps', type=int, default=10, required=False)
    parser.add_argument('--data_directory', type=str, default=None, required=False)
    parser.add_argument('--pretrained_model', type=str, default=None, required=False)
    parser.add_argument('--model_name', type=str, default='trained.h5', required=False)

    args = parser.parse_args()

    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_width=args.image_width, max_txt_length=args.max_txt_length)
    model = AttentionOCR(vocabulary=voc, max_txt_length=args.max_txt_length)

    if not args.data_directory:
        train_data = synthetic_data_generator(vec, epoch_size=args.epoch_size, augment=False, is_training=True)
        validation_data = synthetic_data_generator(vec, epoch_size=args.epoch_size, augment=False)
        test_data = synthetic_data_generator(vec, epoch_size=1, augment=False, is_training=True)
    else:
        train_data = CSVDataSource(vec, args.data_directory, 'train.txt', is_training=True)
        validation_data = CSVDataSource(vec, args.data_directory, 'validation.txt')
        test_data = CSVDataSource(vec, args.data_directory, 'test.txt')

    train_gen = tf.data.Dataset.from_generator(train_data, output_types=(tf.float32, tf.float32, tf.float32))
    validation_gen = tf.data.Dataset.from_generator(validation_data, output_types=(tf.float32, tf.float32, tf.float32))

    if args.pretrained_model:
        model.load(args.pretrained_model)
    model.fit_generator(train_gen, epochs=args.epochs, batch_size=args.batch_size, validation_data=validation_gen, validate_every_steps=args.validate_every_steps)
    if args.model_name:
        model.save(args.model_name)

    for image, decoder_input, decoder_output in test_data():
        txt = voc.one_hot_decode(decoder_output, args.max_txt_length)
        pred = model.predict([image])[0]
        model.visualise([image])
        print(txt, "prediction: ", pred)

# Keras Attention OCR

A stable, readable and maintainable of Attention OCR in Tensorflow's Keras.

This sequence to sequence OCR model aims to provide a stable, readable and maintainable implementation of Attention OCR.
Some of these open source models are implemented in pure Tensorflow.
The Tensorflow API is low-level and changes throughout its many versions.
This makes some of the OCR implementations hard to read and maintain.

## To do

* Make the LSTM block bidirectional
* Name all input and output tensors 
* Make parameters (latent_dim, input height and width, output length) configurable
* Build unit tests
* Show a test case on google colab
* Perform a grid search on best parameters for a toy dataset
* Document the whole API

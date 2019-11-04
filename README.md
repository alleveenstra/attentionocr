# Keras Attention OCR

A stable, readable and maintainable of Attention OCR in Keras.

This sequence to sequence OCR model aims to provide a stable, readable and maintainable implementation of Attention OCR.
Some of the open source models are implemented in a way that readability suffers.
Also, the Tensorflow API changes throughout its many versions.
This makes some of the OCR implementations hard to change and maintain.

## To do

* Name all input and output tensors 
* Make parameters (latent_dim, input height and width, output length) configurable
* Scale the images
* Make image width variable
* Build unit tests
* Show a test case on google colab
* Perform a grid search on best parameters for a toy dataset
* Document the whole API

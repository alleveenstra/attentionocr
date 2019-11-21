# Attention OCR

A clear and maintainable implementation of Attention OCR in Tensorflow 2.0.

This sequence to sequence OCR model aims to provide a clear and maintainable implementation of Attention OCR.
Some of the open source models are implemented in a way that readability suffers.
Also, these implementations are in older Tensorflow versions and its API changes throughout its many versions.
This makes some of the OCR implementations hard to use and maintain.

This repository depends upon the following:

*  Tensorflow 2.0
*  Python 3.6+

## References

This work is based on the following work:

*  [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
*  [Focusing Attention: Towards Accurate Text Recognition in Natural Images
](https://arxiv.org/abs/1709.02054)

## To do

*  Make image height variable
*  Name all input and output tensors 
*  Write unit tests with full coverage
*  Show a test case on google colab
*  Perform a grid search on best parameters for a toy dataset
*  Document the whole API

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/4b136e7c17fb4106a94afa985d03e491)](https://www.codacy.com/manual/alle.veenstra/attentionocr?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=alleveenstra/attentionocr&amp;utm_campaign=Badge_Grade)

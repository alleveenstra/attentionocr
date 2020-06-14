# Attention OCR

A clear and maintainable implementation of Attention OCR in Tensorflow 2.0.

This sequence to sequence OCR model aims to provide a clear and maintainable implementation of attention based OCR.

Please note that this is currently a work in progress.
Documentation is missing, but will be added when the code is stable.

This repository depends upon the following:

*  Tensorflow 2.0
*  Python 3.6+

## Training a model

To train a model, first download the sources for generating synthetic data:

```bash
cd synthetic
./download_data_sources.sh
```

Next, in this project's root folder, run the training script:

```bash
python3 run.py
```

This will run a test training run. 
If everything went well, you'll find a file named "trained.h5" in your directory.
To train a real model you should change the training parameters.
See run.py its arguments to find out what is configurable.

```bash
python3 run.py --help
```

## References

This work is based on the following work:

*  [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## To do

*  Make image height variable
*  Name all input and output tensors 
*  Write unit tests with full coverage
*  Show a test case on google colab
*  Perform a grid search on best parameters for a toy dataset
*  Document the whole API

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/4b136e7c17fb4106a94afa985d03e491)](https://www.codacy.com/manual/alle.veenstra/attentionocr?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=alleveenstra/attentionocr&amp;utm_campaign=Badge_Grade)

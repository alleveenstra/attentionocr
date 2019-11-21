import setuptools

setuptools.setup(
    name='attentionocr',
    version='0.1',
    description='Attention OCR in Tensorflow 2.0',
    author='Alle Veenstra',
    author_email='alle.veenstra@gmail.com',
    url='https://www.github.org/alleveenstra/attentionocr/',
    packages=setuptools.find_packages(include=['attentionocr', 'attentionocr.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

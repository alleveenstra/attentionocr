import os
import random
import logging
import traceback
from glob import glob
from functools import partial
from typing import Optional

from . import Vectorizer


LOG = logging.getLogger(__file__)


def FlatDirectoryDataSource(vectorizer: Vectorizer, glob_pattern: str, max_items: Optional[int] = None, is_training: bool = False):
    images = glob(glob_pattern)
    examples = [(os.path.basename(image_file).split('.')[0], image_file) for image_file in images]
    if max_items is not None:
        random.shuffle(examples)
        examples = examples[:max_items]
    return partial(examples_generator, examples=examples, vectorizer=vectorizer, is_training=is_training)


def CSVDataSource(vectorizer: Vectorizer, directory: str, filename: str, sep: str = ';', is_training: bool = False):
        examples = []
        with open(os.path.join(directory, filename), 'r') as fp:
            for line in fp.readlines():
                if sep in line:
                    image_file, txt = line.split(sep=sep, maxsplit=1)
                    image_file = os.path.abspath(os.path.join(directory, image_file))
                    txt = txt.strip()
                    if os.path.isfile(image_file):
                        examples.append((txt, image_file))
        return partial(examples_generator, examples=examples, vectorizer=vectorizer, is_training=is_training)


def examples_generator(examples, vectorizer, is_training):
    random.shuffle(examples)
    for text, image_file in examples:
        try:
            image = vectorizer.load_image(image_file)
            decoder_input, decoder_output = vectorizer.transform_text(text, is_training)
            yield image, decoder_input, decoder_output
        except Exception as err:
            LOG.warning(err)
            traceback.print_tb(err.__traceback__)

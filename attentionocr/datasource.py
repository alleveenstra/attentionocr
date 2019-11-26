import os
import random
import logging
import traceback
from glob import glob
from functools import partial

from . import Vectorizer


LOG = logging.getLogger(__file__)


def to_json(filename: str):
    pre, _ = os.path.splitext(filename)
    return "%s.json" % pre


def FlatDirectoryDataSource(vectorizer: Vectorizer, glob_pattern: str, is_training: bool = False):
    images = glob(glob_pattern)
    examples = [(os.path.basename(image_file).split('.')[0], image_file, to_json(image_file)) for image_file in images]
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
                        examples.append((txt, image_file, to_json(image_file)))
        return partial(examples_generator, examples=examples, vectorizer=vectorizer, is_training=is_training)


def examples_generator(examples, vectorizer, is_training):
    random.shuffle(examples)
    for text, image_file, focus_file in examples:
        try:
            image = vectorizer.load_image(image_file)
            focus = vectorizer.create_focus(focus_file)
            decoder_input, decoder_output = vectorizer.transform_text(text, is_training)
            yield image, decoder_input, decoder_output, focus
        except Exception as err:
            LOG.warning(err)
            traceback.print_tb(err.__traceback__)

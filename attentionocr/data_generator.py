import string
from glob import glob
from random import randint, choice
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from imgaug import augmenters as iaa

from .image import ImageUtil
from .vectorizer import Vectorizer
from .vocabulary import default_vocabulary

image_util = ImageUtil(32, 320)

seq = iaa.SomeOf((0, 2), [
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    iaa.Invert(1.0),
    iaa.MotionBlur(k=15)
])


def random_font():
    fontname = choice(list(glob('fonts/*.ttf')))
    font = ImageFont.truetype(fontname, size=randint(24, 32))
    return font


def rand_pad():
    return randint(5, 35), randint(5, 35), randint(0, 3), randint(10, 13)


def random_string(length):
    letters = list(string.ascii_uppercase) + default_vocabulary
    return (''.join(choice(letters) for _ in range(length))).strip()


def random_background(height, width):
    background_image = choice(list(glob('images/*.jpg')))
    original = Image.open(background_image)
    L = original.convert('L')
    original = Image.merge('RGB', (L, L, L))
    left = randint(0, original.size[0] - height)
    top = randint(0, original.size[1] - width)
    right = left + height
    bottom = top + width
    return original.crop((left, top, right, bottom))


def generate_image(text: str, augment: bool) -> Tuple[np.array, str, list]:
    font = random_font()
    txt_width, txt_height = font.getsize(text)
    left_pad, right_pad, top_pad, bottom_pad = rand_pad()
    height = left_pad + txt_width + right_pad
    width = top_pad + txt_height + bottom_pad
    image = random_background(height, width)
    stroke_sat = int(np.array(image).mean())
    sat = int((stroke_sat + 127) % 255)
    canvas = ImageDraw.Draw(image)
    canvas.text((left_pad, top_pad), text, fill=(sat, sat, sat), font=font, stroke_width=2, stroke_fill=(stroke_sat, stroke_sat, stroke_sat))
    image = np.array(image)
    if augment:
        image = seq.augment_image(image)
    metadata = []
    for idx, char in enumerate(text):
        char_width, _ = font.getmask(char).size
        x_offset, _ = font.getmask(text[:idx]).size
        metadata.append({'char': char.lower(), 'x': left_pad + x_offset, 'width': char_width})
    image = image_util.preprocess(image)
    return image, text.lower(), metadata


def synthetic_data_generator(vectorizer: Vectorizer, epoch_size: int = 1000, augment: bool = False, is_training: bool = False):

    def synthesize():
        for _ in range(epoch_size):
            image, text, character_positions = generate_image(random_string(randint(4, 20)), augment)
            focus = vectorizer.create_focus(character_positions)
            decoder_input, decoder_output = vectorizer.transform_text(text, is_training)
            yield image, decoder_input, decoder_output, focus

    return synthesize

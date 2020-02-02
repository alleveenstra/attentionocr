import string
from glob import glob
from random import randint, choice
from typing import Tuple, Optional

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
    iaa.MotionBlur(k=10)
])


def random_font():
    fontname = choice(list(glob('fonts/*.ttf')))
    font = ImageFont.truetype(fontname, size=randint(24, 32))
    return font


def rand_pad():
    return randint(5, 35), randint(5, 35), randint(0, 3), randint(10, 13)


def random_string(length: Optional[int] = None):
    if length is None:
        length = randint(4, 20)

    if randint(0, 1) == 0:
        random_file = choice(list(glob('texts/*.txt')))
        with open(random_file, 'r') as f:
            random_txt = f.readlines()
        random_txt = choice(random_txt)
        end = len(random_txt) - length
        if end > 0:
            start = randint(0, end)
            random_txt = random_txt[start:start+length].strip()
            if len(random_txt) > 1:
                return random_txt

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


def generate_image(text: str, augment: bool) -> Tuple[np.array, str]:
    font = random_font()
    txt_width, txt_height = font.getsize(text)
    left_pad, right_pad, top_pad, bottom_pad = rand_pad()
    height = left_pad + txt_width + right_pad
    width = top_pad + txt_height + bottom_pad
    image = random_background(height, width)

    stroke_sat = int(np.array(image).mean())
    sat = int((stroke_sat + 127) % 255)
    mask = Image.new('L', (height, width))
    canvas = ImageDraw.Draw(mask)
    canvas.text((left_pad, top_pad), text, fill=sat, font=font, stroke_fill=stroke_sat, stroke_width=2)
    lower = int(-10 + (txt_width / 32))
    upper = int(10 - (txt_width / 32))
    if upper < lower:
        upper = lower
    mask = mask.rotate(randint(lower, upper))
    image.paste(mask, (0, 0), mask)

    image = np.array(image)
    if augment:
        image = seq.augment_image(image)
    image = image_util.preprocess(image)
    return image, text.lower()


def synthetic_data_generator(vectorizer: Vectorizer, epoch_size: int = 1000, augment: bool = False, is_training: bool = False):

    def synthesize():
        for _ in range(epoch_size):
            image, text = generate_image(random_string(), augment)
            decoder_input, decoder_output = vectorizer.transform_text(text, is_training)
            yield image, decoder_input, decoder_output

    return synthesize

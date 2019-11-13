import json
import random
import string
import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def generate_image(width: int = 144, x_start: int = 0, y_start: int = 0) -> Tuple[np.array, str]:
    n_chars = width // 15
    img = Image.new('RGB', (width, 32), color=(50, 50, 50))
    canvas = ImageDraw.Draw(img)
    font = ImageFont.truetype('Lato-Bold.ttf', size=24)
    txt = random_string(n_chars)
    canvas.text((x_start, y_start), txt, fill='#FFFFFF', font=font)
    metadata = []
    x = x_start
    for char in txt:
        w, h = font.getmask(char).size
        metadata.append({'char': char, 'width': w, 'height': h, 'x': x})
        x += w
    return img, txt, metadata


if __name__ == "__main__":

    if not os.path.isfile("Lato-Bold.ttf"):
        os.system("wget https://github.com/google/fonts/raw/master/ofl/lato/Lato-Bold.ttf")

    if not os.path.exists('train/'):
        os.mkdir('train/')
    if not os.path.exists('test/'):
        os.mkdir('test/')

    for i in range(10000):
        img, txt, meta = generate_image(random.randint(100, 400))
        img.save('train/%s.jpg' % txt)
        with open('train/%s.json' % txt, 'w') as f:
            json.dump(meta, f)

    for i in range(64):
        img, txt, meta = generate_image(random.randint(100, 400))
        img.save('test/%s.jpg' % txt)
        with open('test/%s.json' % txt, 'w') as f:
            json.dump(meta, f)

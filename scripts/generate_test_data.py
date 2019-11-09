import random
import string
import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def generate_image(width=144) -> Tuple[np.array, str]:
    n_chars = width // 15
    img = Image.new('RGB', (width, 32), color=(50, 50, 50))
    canvas = ImageDraw.Draw(img)
    font = ImageFont.truetype('Lato-Bold.ttf', size=24)
    txt = random_string(n_chars)
    canvas.text((0, 0), txt, fill='#FFFFFF', font=font)
    return img, txt


if __name__ == "__main__":

    if not os.path.isfile("Lato-Bold.ttf"):
        os.system("wget https://github.com/google/fonts/raw/master/ofl/lato/Lato-Bold.ttf")

    if not os.path.exists('train/'):
        os.mkdir('train/')
    if not os.path.exists('test/'):
        os.mkdir('test/')

    for i in range(100000):
        img, txt = generate_image(random.randint(100, 400))
        img.save('train/%s.jpg' % txt)

    for i in range(64):
        img, txt = generate_image(random.randint(100, 400))
        img.save('test/%s.jpg' % txt)

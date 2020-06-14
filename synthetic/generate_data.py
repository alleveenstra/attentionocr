import os

from PIL import Image
import numpy as np
from tqdm import tqdm

from attentionocr import generate_image, random_string


def generate_set(directory: str, name: str, size: int, augment: bool):
    if not os.path.exists(f"{directory}/{name}/"):
        os.mkdir(f"{directory}/{name}/")
    with open(f"{directory}/{name}.txt", 'w') as fp:
        for item in tqdm(range(size), desc=name):
            txt = random_string()
            img, txt = generate_image(txt, augment=augment)
            img = np.squeeze(img, axis=-1)
            img = Image.fromarray(np.uint8((img + 1.0) * 127.5))
            img.save(f'{directory}/{name}/{item}.jpg')
            fp.write(f'{name}/{item}.jpg;{txt}\n')


if __name__ == "__main__":
    generate_set('synthetic/data', 'train', 1_000_000, True)
    generate_set('synthetic/data', 'validation', 256, False)
    generate_set('synthetic/data', 'test', 256, False)

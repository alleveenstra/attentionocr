from random import randint

import numpy as np

from attentionocr.data_generator import generate_image, random_string


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    string = random_string(randint(4, 20))
    print(string)
    img, txt = generate_image(string, True)
    print(txt)
    img = np.squeeze(img, -1)
    imgplot = plt.imshow(img)
    plt.show()

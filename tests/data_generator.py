from random import randint


from attentionocr.data_generator import generate_image, random_string


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img, txt, meta = generate_image(random_string(randint(4, 20)), True)
    imgplot = plt.imshow(img)
    plt.show()

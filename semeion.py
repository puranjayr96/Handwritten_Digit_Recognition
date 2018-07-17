import numpy
from random import shuffle

def read_data_semeion(fname='semeion/semeion.data.txt'):
    file = open(fname, 'r')
    lines = file.readlines()

    width = 16
    height = 16
    size = width * height
    classes = 10

    images = []
    labels = []

    for line in lines:
        data = line.split(' ')
        image = []
        label = []

        for i in range(0, size):
            image.append(int(float(data[i])))
        images.append(image)

        for i in range(size, size + classes):
            label.append(int(float(data[i])))
        labels.append(label)

    # Shuffle data
    images_shuffle = []
    labels_shuffle = []
    indexes = list(range(len(images)))
    shuffle(indexes)
    for i in indexes:
        images_shuffle.append(images[i])
        labels_shuffle.append(labels[i])

    images = images_shuffle
    labels = labels_shuffle

    images = numpy.array(images,dtype=numpy.uint8)
    labels = numpy.array(labels, dtype=numpy.uint8)

    return images, labels
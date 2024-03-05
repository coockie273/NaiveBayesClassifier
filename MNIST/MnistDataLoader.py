# Testing accuracy of the naive Bayes classification with dataset "MNIST"

import struct
import numpy as np
from array import array
from src.naiveBayesClassifier import NaiveBayesClassifier
from src.utils import cross_validation_accuracy


def read_images_labels(images_filepath, labels_filepath):
    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())

    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img

    return images, labels


# Import dataset
training_images_filepath = 'train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'


x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)


mnist_train_data = []
for i in range(60000):
    data = []
    for raw in x_train[i]:
        data.extend(raw)
    data.append(y_train[i])
    mnist_train_data.append(data)

# Initialization model
model = NaiveBayesClassifier(784, window_size=5)

# Training
model.train(mnist_train_data)

# Calculation accuracy with Shuffle-Split Cross-Validation method
print(cross_validation_accuracy(model, mnist_train_data))



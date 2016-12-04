#! /usr/bin/python3

import numpy as np

class MnistTrainingSet:

    def __init__(self):
        mnist_training_images_filename = "../data/train-images-idx3-ubyte"
        self.f = open(mnist_training_images_filename, "rb")

        self.f.seek(4)
        self.N_images = int.from_bytes(self.f.read(4), byteorder="big")

        self.f.seek(8)
        self.rows = int.from_bytes(self.f.read(4), byteorder="big")

        self.f.seek(12)
        self.cols = int.from_bytes(self.f.read(4), byteorder="big")

    def getImageAsBytes(self, index):
        if (index < 1 or index > self.N_images):
            return None
        else:
            self.f.seek(16 + (index-1)*self.rows*self.cols)
            return self.f.read(self.rows*self.cols)

    def getImageAsFloatArray(self, index):
        return np.reshape([float(x)/255.0 for x in self.getImageAsBytes(index)],
                          (28, 28))

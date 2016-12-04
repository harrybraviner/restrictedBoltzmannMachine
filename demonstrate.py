#! /usr/bin/python3

import numpy as np
import matplotlib.image as imgplot
import matplotlib.pyplot as plt
import tensorflow as tf

from mnist import MnistTrainingSet

def displayTrainingImages(mnist, indices):
    n1 = 20
    n2 = 20
    if (len(indices) != n1*n2):
        raise Exception("Need 10 indices")
    else:
        images = [mnist.getImageAsFloatArray(index) for index in indices]
        images = [tf.cast(tf.constant(image)*255, dtype=tf.uint8) for image in images]
        combined_image = tf.concat(0, [tf.concat(1, [images[i] for i in range(5*j, 5*j+n2)]) for j in range(n1)])
        combined_image = tf.pack([combined_image, combined_image, combined_image], axis=2)
        with tf.Session() as session:
            result = session.run(combined_image)
        plt.imshow(result)
        plt.show()




def main():
    mnist = MnistTrainingSet()

    print(mnist.N_images)

    #plt.imshow(mnist.getImageAsFloatArray(3))
    #plt.show()

    indicesToShow = np.random.choice(mnist.N_images, 20*20)
    displayTrainingImages(mnist, indicesToShow)

if (__name__ == "__main__"):
    main()

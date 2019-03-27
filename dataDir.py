import os
import sys
import argparse
import pandas as pd

from mobilenet_v2 import MobileNetv2

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

def generate(batch, size=32):
    """Data generation and augmentation
    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.
    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    ptrain = 'data224/train'
    pval = 'data224/test'

    datagen1 = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(samplewise_center=True,
        samplewise_std_normalization=True,)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1

    return train_generator, validation_generator, count1, count2

from __future__ import print_function

import cv2         # working with, mainly resizing, images
import numpy as np     # dealing with arrays
import os              # dealing with directories
from tqdm import tqdm	   # percentage bar for tasks
import random
import matplotlib.pyplot as plt
from matplotlib import ticker
# %matplotlib inline

from config import Config
import data_utils as du

config = Config()

TRAIN_DIR = config.train_dir
TEST_DIR = config.test_dir
IMG_SIZE = config.image_size
CHANNELS = config.channels
LR = config.lr


def label_img(img):
    '''
    Images are labeled as "cat.1.jpg" or "dog.3.jpg" and so on, so we can just
    split out the dog/cat, and then convert to an array like so:
    '''
    label = img.split('.')[-3]

    # conversion to one-hot array [cat,dog]
    if label == 'cat':
        return [1, 0]
    elif label == 'dog':
        return [0, 1]


def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # image normalization
        # mean_pixel = [103.939, 116.779, 123.68]
        # img = img.astype(np.float32, copy=False)
        # for c in range(3):
        #     img[:, :, c] = img[:, :, c] - mean_pixel[c]
        img = img.transpose((2, 0, 1))
        # img = np.expand_dims(img, axis=0)

        train_data.append([np.array(img), np.array(label)])

    np.random.shuffle(train_data)
    # np.save('train_data.npy', train_data, fix_imports=True)
    return np.array(train_data)


def process_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.transpose((2, 0, 1))
        test_data.append([np.array(img), img_num])

    np.save('test_data.npy', test_data)
    return np.array(test_data)


def process_data():
    if os.path.exists('train_data.npy') and os.path.exists('validation_data.npy'):
        print("\nLoading existing training data")
        train_data = np.load('train_data.npy')
        print("Loading existing validation data")
        validation_data = np.load('validation_data.npy')

        print("Shuffling and Re-splitting into train/validation data set")
        train_data, validation_data = \
            du.split_dataset(train_data, validation_data, config.split_rate)
    else:
        print("Creating training data")
        train_data = create_train_data()
        print("Splitting into train/validation data set")
        train_data, validation_data = \
            du.split_dataset(train_data, None, config.split_rate)

    if os.path.exists('test_data.npy'):
        print("Loading existing test data")
        test_data = np.load('test_data.npy')
        print(test_data.shape)
    else:
        print("Processing test data")
        test_data = process_test_data()
        print(test_data.shape)

    return train_data, validation_data, test_data


if __name__ == '__main__':

    train_data, validation_data, test_data = process_data()

    batches = generate_train_batches(train_data, config.batch_size)
    next = get_next_batch(batches)
    print(len(batches))
    print(batches[0][0][0].shape)

from __future__ import print_function

import numpy as np     # dealing with arrays
import os              # dealing with directories
from tqdm import tqdm	   # percentage bar for tasks
import random
import matplotlib.pyplot as plt
from matplotlib import ticker
# %matplotlib inline

from config import Config

config = Config()

TRAIN_DIR = config.train_dir
TEST_DIR = config.test_dir
IMG_SIZE = config.image_size
CHANNELS = config.channels
LR = config.lr


def split_dataset(train_data, validation_data, split_rate):
    if validation_data is not None:
        data = np.concatenate([train_data, validation_data])
    else:
        data = train_data

    np.random.shuffle(data)
    print(data.shape)
    train_data = data[:-5000]
    validation_data = data[-5000:]

    print(train_data.shape)
    print(validation_data.shape)

    if not os.path.exists('validation_data.npy'):
        np.save('train_data.npy', train_data)
        np.save('validation_data.npy', validation_data)

    return train_data, validation_data


def generate_train_batches(train_data, batch_size):
    print("Generating training batches")
    batches = np.array_split(train_data, len(train_data) / batch_size)
    return random.sample(batches, len(batches))


def get_next_batch(batches):
    # print("Popping next batch")
    next = batches.pop(0)
    return next


def load_data():
    print("\nLoading existing training data")
    train_data = np.load('train_data.npy')
    print("Loading existing validation data")
    validation_data = np.load('validation_data.npy')
    print("Loading existing test data")
    test_data = np.load('test_data.npy')
    print("Shuffling and Re-splitting into train/validation data set")
    train_data, validation_data = \
        split_dataset(train_data, validation_data, config.split_rate)
    return train_data, validation_data, test_data


if __name__ == '__main__':

    import image_utils as iu
    train_data, validation_data, test_data = iu.process_data()

    batches = generate_train_batches(train_data, config.batch_size)
    next = get_next_batch(batches)
    print(len(batches))
    print(batches[0][0][0].shape)
    # fig = plt.figure()

    # np.random.shuffle(train_data)
    # index = 1
    # for data in train_data[:12]:
    #     print(data[1])
    #     y = fig.add_subplot(3, 4, index)
    #     if data[1][0]:
    #         label = "cat"
    #     else:
    #         label = "dog"
    #     y.imshow(data[0], cmap='gray')
    #     plt.title(label)
    #     y.axes.get_xaxis().set_visible(False)
    #     y.axes.get_yaxis().set_visible(False)
    #     index += 1

    # plt.show()

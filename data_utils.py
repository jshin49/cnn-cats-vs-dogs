from __future__ import print_function

import numpy as np     # dealing with arrays
import os              # dealing with directories
from tqdm import tqdm	   # percentage bar for tasks
import random
import matplotlib.pyplot as plt
from matplotlib import ticker
import tensorflow as tf
# %matplotlib inline

from config import Config

config = Config()

TRAIN_DIR = config.train_dir
TEST_DIR = config.test_dir
IMG_DIR = config.image_dir
IMG_SIZE = config.image_size
CHANNELS = config.channels
LR = config.lr


def split_dataset(train_data, validation_data, size=IMG_SIZE):
    if validation_data is not None:
        data = np.concatenate([train_data, validation_data])
    else:
        data = train_data

    np.random.shuffle(data)
    print(data.shape)
    train_data = data[:-1000]
    validation_data = data[-1000:]

    print(train_data.shape)
    print(validation_data.shape)

    if not os.path.exists(IMG_DIR + 'validation_data' + str(size) + '.npy'):
        np.save(IMG_DIR + 'train_data' + str(size) + '.npy', train_data)
        np.save(IMG_DIR + 'validation_data' +
                str(size) + '.npy', validation_data)

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
    train_data = np.load(IMG_DIR + 'train_data' + str(IMG_SIZE) + '.npy')
    print("Loading existing validation data")
    validation_data = np.load(
        IMG_DIR + 'validation_data' + str(IMG_SIZE) + '.npy')
    print("Loading existing test data")
    test_data = np.load(IMG_DIR + 'test_data' + str(IMG_SIZE) + '.npy')
    print("Shuffling and Re-splitting into train/validation data set")
    train_data, validation_data = \
        split_dataset(train_data, validation_data, config.split_rate)
    return train_data, validation_data, test_data


if __name__ == '__main__':

    import image_utils as iu
    for size in [64, 150, 224]:
        train_data, validation_data, test_data = iu.process_data(size)

    train_batches = generate_train_batches(train_data, 12)
    print(len(train_batches))
    print(train_batches[0][0][0].shape)
    val_batches = generate_train_batches(
        validation_data, config.batch_size)
    print(len(val_batches))

    # b1 = len(train_batches)
    # b2 = len(val_batches)
    # for step in tqdm(range(b1)):
    #     if step % (b1 / b2) == 0 and b2 * (b1 / b2) <= step:
    #         # print(step)
    #         # train_batch = get_next_batch(train_batches)
    #         val_batch = get_next_batch(val_batches)

    # train_batches = tf.train.batch(map(list, train_data), config.batch_size)
    # print(train_batches.shape)
    # fig = plt.figure()

    # index = 1
    # for data in get_next_batch(train_batches):
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

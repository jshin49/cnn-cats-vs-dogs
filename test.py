from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm	   # percentage bar for tasks

from model import Model
from config import Config
from data_utils import load_data, generate_train_batches, get_next_batch

# 0=Test, 1=Train
K.set_learning_phase(0)

# Initialize model
graph = tf.Graph()
sess_config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

config = Config()
model = Model(config, sess, graph)

model.restore()

# Generate data and batches for each epoch
train_data, validation_data, test_data = load_data()
# train_images, train_labels = map(list, zip(*train_data))
# test_images, test_labels = map(list, zip(*test_data))

val_batches = generate_train_batches(validation_data, 100)
val_batch = get_next_batch(val_batches)
validation_images, validation_labels = map(list, zip(*val_batch))

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)
validation_images = validation_images.reshape(-1, config.image_size,
                                              config.image_size, config.channels)

pred, acc = model.eval_batch(validation_images, validation_labels)

print(acc)

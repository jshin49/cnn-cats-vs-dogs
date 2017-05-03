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
K.set_image_dim_ordering('th')

# Initialize model
graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

config = Config()
model = Model(config, sess, graph)

model.restore()

# Generate data and batches for each epoch
train_data, validation_data, test_data = load_data()

train_batches = generate_train_batches(train_data, 100)
train_batch = get_next_batch(train_batches)
train_batch_images, train_batch_labels = map(list, zip(*train_batch))
train_batch_images = np.array(train_batch_images)
train_batch_labels = np.array(train_batch_labels)
train_batch_images = train_batch_images.reshape(-1, config.channels,
                                                config.image_size, config.image_size)

acc1 = model.eval_batch(train_batch_images, train_batch_labels)
print(acc1)

val_batches = generate_train_batches(validation_data, 100)
val_batch = get_next_batch(val_batches)
val_batch_images, val_batch_labels = map(list, zip(*val_batch))
val_batch_images = np.array(val_batch_images)
val_batch_labels = np.array(val_batch_labels)
val_batch_images = val_batch_images.reshape(-1, config.channels,
                                            config.image_size, config.image_size)

acc2 = model.eval_batch(val_batch_images, val_batch_labels)
print(acc2)
print('Training Acc: %f, Validation Acc: %f' % (acc1, acc2))

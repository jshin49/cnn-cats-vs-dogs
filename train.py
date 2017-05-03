from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm	   # percentage bar for tasks

from model import Model
from config import Config
from data_utils import load_data, generate_train_batches, get_next_batch


def train(train_data, validation_data, total_batch_size, val_batch_size):
    for epoch in tqdm(range(config.epochs)):
        avg_loss = 0
        avg_acc = 0
        avg_val_loss = 0
        avg_val_acc = 0
        train_batches = generate_train_batches(train_data, config.batch_size)
        val_batches = generate_train_batches(
            validation_data, 100)

        for step in tqdm(range(total_batch_size)):
            train_batch = get_next_batch(train_batches)

            train_batch_images, train_batch_labels = map(
                list, zip(*train_batch))
            train_batch_images = np.array(train_batch_images)
            train_batch_labels = np.array(train_batch_labels)
            train_batch_images = train_batch_images.reshape(-1, config.image_size,
                                                            config.image_size, config.channels)

            loss, acc = model.train_eval_batch(
                train_batch_images, train_batch_labels)
            avg_loss += loss / total_batch_size
            avg_acc += acc / total_batch_size

            if step % (total_batch_size / val_batch_size) == 0:
                val_batch = get_next_batch(val_batches)
                val_batch_images, val_batch_labels = map(list, zip(*val_batch))
                val_batch_images = np.array(val_batch_images)
                val_batch_labels = np.array(val_batch_labels)
                val_batch_images = val_batch_images.reshape(-1, config.image_size,
                                                            config.image_size, config.channels)

                val_loss, val_acc = model.eval_batch(
                    val_batch_images, val_batch_labels)
                avg_val_loss += val_loss / val_batch_size
                avg_val_acc += val_acc / val_batch_size

        print('\nEpoch: %d, Avg. Loss: %f, Val Loss: %f' %
              (epoch, avg_loss, avg_val_loss))
        print('\nEpoch: %d, Train Acc: %f, Val Acc: %f' %
              (epoch, avg_acc, avg_val_acc))
        print('saving checkpoint')
        model.save((epoch + 1) * total_batch_size)

    print('Training Completed')


# Initialize model
graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
# sess = tf.Session()
config = Config()
model = Model(config, sess, graph)

# Generate data and batches for each epoch
train_data, validation_data, test_data = load_data()
val_batch_size = int(len(validation_data) / 100)

# Hyperparameter Tuning (Choose Best)
l2s = [0.01]
lrs = [0.005, 0.001, 0.0005, 0.0001]
dropouts = [0.5, 0.75, 1.0]
batch_sizes = [32, 64, 128]
for l2 in l2s:
    config.l2 = l2
    for dropout in dropouts:
        config.dropout = dropout
        for batch_size in batch_sizes:
            config.batch_size = batch_size
            total_batch_size = int(config.train_size / config.batch_size)
            for lr in lrs:
                config.lr = lr
                model.restore()
                print('L2: %f, Dropout: %f, Batch Size: %d, Learning Rate: %f \n' %
                      (config.l2, config.dropout, config.batch_size, config.lr))
                train(train_data, validation_data,
                      total_batch_size, val_batch_size)

# model.restore()
# train(train_data, validation_data, total_batch_size, val_batch_size)

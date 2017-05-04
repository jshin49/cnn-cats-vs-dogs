from __future__ import print_function

import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tqdm import tqdm	   # percentage bar for tasks
import random

from model import Model
from config import Config
from data_utils import load_data, generate_train_batches, get_next_batch


def train(train_data, total_batch_size, validation_data=None):
    model.sess.run(model.init)

    for epoch in tqdm(range(config.epochs)):
        avg_loss = 0
        avg_acc = 0
        train_batches = generate_train_batches(train_data, config.batch_size)

        for step in tqdm(range(total_batch_size)):
            train_batch = get_next_batch(train_batches)

            train_batch_images, train_batch_labels = map(
                list, zip(*train_batch))
            train_batch_images = np.array(train_batch_images).reshape(-1, config.image_size,
                                                                      config.image_size, config.channels)
            train_batch_labels = np.array(train_batch_labels).reshape(-1, 1)

            summary, loss, acc = model.train_eval_batch(
                train_batch_images, train_batch_labels, False)
            avg_loss += (loss / total_batch_size)
            avg_acc += (acc / total_batch_size)
            model.writer.add_summary(
                summary, (epoch + 1) * total_batch_size + step)

        print('\nEpoch: %d, Avg Loss: %f, Train Acc: %f' %
              (epoch + 1, avg_loss, avg_acc))

        val_images, val_labels = map(
            list, zip(*random.sample(validation_data, 100)))
        val_images = np.array(val_images).reshape(-1, config.image_size,
                                                  config.image_size, config.channels)
        val_labels = np.array(val_labels).reshape(-1, 1)
        val_loss, val_acc = model.eval_batch(val_images, val_labels)
        print('\nEpoch: %d, Validation Loss: %f, Validation Acc: %f' %
              (epoch + 1, val_loss, val_acc))

        print('saving checkpoint')
        model.save((epoch + 1) * total_batch_size)

    print('Training Completed')
    model.writer.close()


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

# Hyperparameter Tuning (Choose Best)
image_sizes = [64, 150, 224]
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
l2s = [0.1, 0.01, 0.001, 0.0001, 0.00001]
lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
dropouts = [0.5, 0.75, 1.0]
batch_sizes = [16, 32, 64, 100, 128]

# for dropout in dropouts:
#     pass

# for threshold in thresholds:
#     pass

for lr in lrs:
    config.lr = lr
    model.restore()
    # print('L2: %f, Dropout: %f, Batch Size: %d, Learning Rate: %f \n' %
    #       (config.l2, config.dropout, config.batch_size, config.lr))
    print('Learning Rate: %f' % (lr))
    total_batch_size = int(config.train_size / config.batch_size)
    train(train_data, total_batch_size, validation_data)


# for batch_size in batch_sizes:
#     config.batch_size = batch_size
#     model.restore()
#     print('Batch Size: %f' % (batch_size))
#     total_batch_size = int(config.train_size / config.batch_size)
#     train(random.sample(train_data, model.config.train_size), total_batch_size,
#           random.sample(validation_data, model.config.valid_size))

# for threshold in thresholds:
#     config.threshold = threshold
#     model.restore()
#     print('Threshold: %f' % (threshold))
#     total_batch_size = int(config.train_size / config.batch_size)
#     train(random.sample(train_data, model.config.train_size), total_batch_size,
#           random.sample(validation_data, model.config.valid_size))


total_batch_size = int(config.train_size / config.batch_size)
model.restore()
train(train_data, total_batch_size, validation_data)

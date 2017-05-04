from __future__ import print_function

import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tqdm import tqdm	   # percentage bar for tasks
import random

from mnist import Model
from config import Config


def train(train_data, total_batch_size, validation_data=None):
    model.sess.run(model.init)

    for epoch in tqdm(range(config.epochs)):
        avg_loss = 0
        avg_acc = 0

        for step in tqdm(range(total_batch_size)):
            train_batch_images, train_batch_labels = train_data.next_batch(128)

            summary, loss, acc = model.train_eval_batch(np.reshape(
                train_batch_images, [-1, 28, 28, 1]), train_batch_labels, True)
            avg_loss += (loss / total_batch_size)
            avg_acc += (acc / total_batch_size)
            model.writer.add_summary(
                summary, (epoch + 1) * total_batch_size + step)

        print('\nEpoch: %d, Avg Loss: %f, Train Acc: %f' %
              (epoch + 1, avg_loss, avg_acc))

        val_images, val_labels = random.sample(validation_data, 100)
        val_images = np.array(val_images).reshape(-1, config.image_size,
                                                  config.image_size, config.channels)
        val_loss, val_acc = model.eval_batch(val_images, val_labels)
        print('\nEpoch: %d, Validation Loss: %f, Validation Acc: %f' %
              (epoch + 1, val_loss, val_acc))

    print('Training Completed')
    print('Testing')


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

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
total_batch_size = int(mnist.train.num_examples / 128)
train(mnist.train, total_batch_size, mnist.validation)

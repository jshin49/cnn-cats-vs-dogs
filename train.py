from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm	   # percentage bar for tasks

from model import Model
from config import Config
from data_utils import load_data, generate_train_batches, get_next_batch

# 0=Test, 1=Train
K.set_learning_phase(1)

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
total_batch_size = int(config.train_size / config.batch_size)
for epoch in tqdm(range(config.epochs)):
    avg_loss = 0
    avg_val_loss = 0
    train_batches = generate_train_batches(train_data, config.batch_size)
    val_batches = generate_train_batches(
        validation_data, 100)

    for step in tqdm(range(total_batch_size)):
        train_batch = get_next_batch(train_batches)

        train_batch_images, train_batch_labels = map(list, zip(*train_batch))
        train_batch_images = np.array(train_batch_images)
        train_batch_labels = np.array(train_batch_labels)
        train_batch_images = train_batch_images.reshape(-1, config.image_size,
                                                        config.image_size, config.channels)

        loss = model.train_batch(train_batch_images, train_batch_labels)
        avg_loss += loss / total_batch_size

        if step % (total_batch_size / len(val_batches)) == 0:
            val_batch = get_next_batch(val_batches)
            val_batch_images, val_batch_labels = map(list, zip(*val_batch))
            val_batch_images = np.array(val_batch_images)
            val_batch_labels = np.array(val_batch_labels)
            val_batch_images = val_batch_images.reshape(-1, config.image_size,
                                                        config.image_size, config.channels)

            _, val_loss = model.predict(val_batch_images, val_batch_labels)
            avg_val_loss += val_loss / (len(val_batches))

    print('Epoch: %d, Avg. Loss: %f, Val Loss: %f' %
          (epoch, avg_loss, avg_val_loss))
    print('saving checkpoint')
    model.save((epoch + 1) * total_batch_size)

print('Training Completed')
# print('Testing')

# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

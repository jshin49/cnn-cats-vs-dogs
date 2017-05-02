from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm	   # percentage bar for tasks

from model import Model
from config import Config
from data_utils import process_data, generate_train_batches, get_next_batch

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
train_data, validation_data, test_data = process_data()

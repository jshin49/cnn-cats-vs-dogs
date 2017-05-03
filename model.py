import sys
import numpy as np
import tensorflow as tf

from config import Config
import data_utils as du


class Model(object):

    def __init__(self, config, session, graph):
        print("Init Model object")
        self.graph = graph
        self.sess = session
        # K.set_session(self.sess)

        self.config = config
        self.learning_rate = self.config.lr
        self.batch_size = self.config.batch_size
        self.image_size = self.config.image_size
        self.epochs = self.config.epochs
        sys.stdout.write('<log>Building Graph')
        # build computation graph
        self.build_graph()
        sys.stdout.write('</log>\n')

    def init_model(self, images, training):
        # CNN Model with tf.layers

        # Input Layer
        input_layer = tf.reshape(
            images, [-1,
                     self.config.image_size,
                     self.config.image_size,
                     self.config.channels])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        conv1 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[2, 2], strides=(2, 2))

        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[2, 2], strides=(2, 2))

        # Convolutional Layer #3
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(
            inputs=conv1,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[2, 2], strides=(2, 2))

        # Convolutional Layer #4
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(
            inputs=conv1,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[2, 2], strides=(2, 2))

        # Dense Layer
        flatten = tf.reshape(pool4, [-1, 32 * 32 * 32])
        fc1 = tf.layers.dense(
            inputs=flatten, units=256, activation=tf.nn.relu)
        fc1 = tf.layers.dropout(
            inputs=fc1, rate=self.config.dropout, training=training)
        fc2 = tf.layers.dense(
            inputs=fc1, units=256, activation=tf.nn.relu)
        fc2 = tf.layers.dropout(
            inputs=fc2, rate=self.config.dropout, training=training)

        logits = tf.layers.dense(inputs=fc2, units=2)

        return logits

    # build the graph
    def build_graph(self):
        with tf.device('/gpu:0'):
            with self.graph.as_default():
                with self.sess:
                    # Input images
                    self.images = tf.placeholder(shape=[None,
                                                        self.config.image_size,
                                                        self.config.image_size,
                                                        self.config.channels],
                                                 dtype=tf.float32,
                                                 name='Images')

                    # Input labels that represent the real outputs
                    self.labels = tf.placeholder(shape=[None, 2],
                                                 dtype=tf.float32,
                                                 name='Labels')

                    # Is Training?
                    self.training = tf.placeholder(dtype=tf.bool)

                    self.model = self.init_model(self.images, self.training)
                    self.loss = tf.losses.softmax_cross_entropy(
                        onehot_labels=self.labels, logits=self.model)

                    self.optimizer = tf.train.RMSPropOptimizer(
                        learning_rate=self.learning_rate).minimize(self.loss)

                    correct_prediction = tf.equal(
                        tf.argmax(self.model, 1), tf.argmax(self.labels, 1))
                    self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32))

                    self.init = tf.global_variables_initializer()
                    self.saver = tf.train.Saver(tf.trainable_variables())

    def predict(self, batch_images, batch_labels):
        self.sess.run(self.init)
        feed_dict = {
            self.images: batch_images,
            self.labels: batch_labels,
            self.training: False
        }
        pred, loss, acc = self.sess.run(
            [self.model, self.loss, self.accuracy], feed_dict=feed_dict)
        return pred, loss, acc

    def train_eval_batch(self, batch_images, batch_labels):
        self.sess.run(self.init)
        feed_dict = {
            self.images: batch_images,
            self.labels: batch_labels,
            self.training: True
        }
        loss, acc, _ = self.sess.run(
            [self.loss, self.accuracy, self.optimizer], feed_dict=feed_dict)
        return loss, acc

    def eval_batch(self, batch_images, batch_labels):
        self.sess.run(self.init)
        feed_dict = {
            self.images: batch_images,
            self.labels: batch_labels,
            self.training: False
        }
        loss, acc = self.sess.run(
            [self.loss, self.accuracy], feed_dict=feed_dict)
        return loss, acc

    def test_batch(self, batch_images):
        self.sess.run(self.init)
        feed_dict = {
            self.images: batch_images,
            self.training: False
        }
        pred = self.sess.run(
            [self.model], feed_dict=feed_dict)
        return pred

    def save(self, step):
        self.saver.save(self.sess, self.config.ckpt_path +
                        '.ckpt', global_step=step)

    def restore(self):
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.config.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

if __name__ == '__main__':
    graph = tf.Graph()
    sess_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    # sess = tf.Session()
    config = Config()
    model = Model(config, sess, graph)

    train_data, validation_data, test_data = du.load_data()
    batches = du.generate_train_batches(train_data, config.batch_size)
    batch = du.get_next_batch(batches)
    batch_images, batch_labels = map(list, zip(*batch))
    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)
    batch_images = batch_images.reshape(-1, config.image_size, config.image_size,
                                        config.channels)
    pred, loss, acc = model.predict(batch_images, batch_labels)
    print(loss)
    print(pred.shape)

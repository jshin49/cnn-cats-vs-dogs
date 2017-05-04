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
        self.log_path = '/tmp/tensorboard/'
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
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(self.config.l2)

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
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            activation=tf.nn.relu)
        conv1 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[2, 2], strides=(2, 2))

        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(
            inputs=conv2,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=[2, 2], strides=(2, 2))

        # Convolutional Layer #3
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(
            inputs=conv3,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(
            inputs=conv3, pool_size=[2, 2], strides=(2, 2))

        # Convolutional Layer #4
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(
            inputs=conv4,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(
            inputs=conv4, pool_size=[2, 2], strides=(2, 2))
        # return pool2

        # Dense Layer
        # Flatten for 64*64 : 4,4,256
        flatten = tf.reshape(pool4, [-1, 4 * 4 * 256])
        # Flatten for 150*150 : 9,9,256
        # flatten = tf.reshape(pool4, [-1, 9 * 9 * 256])
        # Flatten for 224*224 : 14,14,256
        # flatten = tf.reshape(pool4, [-1, 14 * 14 * 256])
        # Dense Layer

        # Flatten for 64*64 : 16,16,64
        # flatten = tf.reshape(pool2, [-1, 16 * 16 * 64])
        # Flatten for 150*150 : 37,37,64
        # flatten = tf.reshape(pool4, [-1, 37 * 37 * 64])
        # Flatten for 224*224 : 56,56,64
        # flatten = tf.reshape(pool4, [-1, 56 * 56 * 64])
        fc1 = tf.layers.dense(
            inputs=flatten,
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer)
        fc1 = tf.layers.dropout(
            inputs=fc1,
            rate=self.config.dropout,
            training=training)
        # fc2 = tf.layers.dense(
        #     inputs=fc1,
        #     units=256,
        #     activation=tf.nn.relu,
        #     kernel_initializer=initializer,
        #     kernel_regularizer=regularizer)
        # fc2 = tf.layers.dropout(
        #     inputs=fc2,
        #     rate=self.config.dropout,
        #     training=training)

        # One output: Confidence score of being a dog
        logits = tf.layers.dense(inputs=fc1, units=1, activation=tf.nn.sigmoid)

        return logits

    # build the graph
    def build_graph(self):
        with self.graph.as_default():
            with self.sess:
                with tf.device('/gpu:0'):
                    # Input images
                    self.images = tf.placeholder(shape=[None,
                                                        self.config.image_size,
                                                        self.config.image_size,
                                                        self.config.channels],
                                                 dtype=tf.float32,
                                                 name='Images')

                    # Input labels that represent the real outputs
                    self.labels = tf.placeholder(shape=[None, 1],
                                                 dtype=tf.float32,
                                                 name='Labels')

                    # Is Training?
                    self.training = tf.placeholder(dtype=tf.bool)

                    self.model = self.init_model(self.images, self.training)
                    thresholds = tf.fill(
                        [self.config.batch_size], self.config.threshold)
                    self.predictions = tf.greater_equal(
                        self.model, thresholds)
                    correct_prediction = tf.equal(
                        self.predictions, tf.cast(self.labels, tf.bool))
                    self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32))
                    self.loss = tf.losses.log_loss(
                        labels=self.labels, predictions=self.model)
                    # self.accuracy = tf.constant(1)
                    # self.loss = tf.constant(1)
                    self.optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate).minimize(self.loss)

                    # TensorBoard Summary
                    tf.summary.scalar("log_loss", self.loss)
                    tf.summary.scalar("accuracy", self.accuracy)
                    self.summary = tf.summary.merge_all()

                    self.init = tf.global_variables_initializer()
                    self.writer = tf.summary.FileWriter(
                        self.log_path, graph=self.sess.graph_def)

                with tf.device('/cpu:0'):
                    self.saver = tf.train.Saver(tf.trainable_variables())

    def generate_feed_dict(self, batch_images, batch_labels=None, training=False):
        return {
            self.images: batch_images,
            self.labels: batch_labels,
            self.training: training
        }

    # def calculate_accuracy(self, pred, labels):
    #     predictions = map(
    #         lambda x: 1 if x >= self.config.threshold else 0, pred)
    #     correct_prediction = tf.equal(
    #         predictions, labels)
    #     acc = tf.reduce_mean(
    #         tf.cast(correct_prediction, tf.float32))
    #     return self.sess.run(acc)

    def predict(self, batch_images, batch_labels):
        self.sess.run(self.init)
        feed_dict = self.generate_feed_dict(batch_images, batch_labels, False)
        pred, loss, acc = self.sess.run(
            [self.model, self.loss, self.accuracy], feed_dict=feed_dict)
        return pred, loss, acc

    def train_eval_batch(self, batch_images, batch_labels, training=True):
        feed_dict = self.generate_feed_dict(
            batch_images, batch_labels, training)
        summary, loss, acc, _ = self.sess.run(
            [self.summary, self.loss, self.accuracy, self.optimizer], feed_dict=feed_dict)

        return summary, loss, acc

    def eval_batch(self, batch_images, batch_labels, training=False):
        feed_dict = self.generate_feed_dict(
            batch_images, batch_labels, training)
        loss, acc = self.sess.run(
            [self.loss, self.accuracy], feed_dict=feed_dict)
        return loss, acc

    def test_batch(self, batch_images):
        self.sess.run(self.init)
        feed_dict = self.generate_feed_dict(batch_images)
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
    print(batch_labels)
    batch_labels = np.array(batch_labels).reshape(-1, 1)
    batch_images = batch_images.reshape(-1, config.image_size,
                                        config.image_size, config.channels)
    pred, loss, acc = model.predict(batch_images, batch_labels)
    # zeros = np.zeros(
    #     (16, config.image_size, config.image_size, 3), dtype=np.int)
    # pred, loss, acc = model.predict(zeros, batch_labels)
    print(pred, batch_labels)
    print(pred.shape)
    print(loss)
    print(acc)

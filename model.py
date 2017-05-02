import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.objectives import categorical_crossentropy

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

    def init_model(self, images):
        # CNN Model
        conv1 = Conv2D(32, (3, 3), padding='same', input_shape=(self.config.image_size,
                                                                self.config.image_size,
                                                                self.config.channels), activation='relu')(images)
        conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
        conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
        conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
        conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
        conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
        conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
        conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv4)
        conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        features = Flatten()(conv4)

        fc1 = Dense(256, activation='relu')(features)
        fc1 = Dropout(self.config.dropout)(fc1)
        fc2 = Dense(256, activation='relu')(fc1)
        fc2 = Dropout(self.config.dropout)(fc2)
        out = Dense(2, activation='softmax')(fc2)

        return out

    # build the graph
    def build_graph(self):
        tf.reset_default_graph()
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

                self.model = self.init_model(self.images)
                self.loss = tf.reduce_mean(
                    categorical_crossentropy(self.labels, self.model))

                self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss)
                correct_prediction = tf.equal(
                    tf.argmax(self.labels, 1), tf.argmax(self.model, 1))

                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(tf.trainable_variables())

    def predict(self, batch_images, batch_labels):
        K.set_learning_phase(0)
        self.sess.run(self.init)
        feed_dict = {
            self.images: batch_images,
            self.labels: batch_labels
        }
        pred, loss = self.sess.run(
            [self.model, self.loss], feed_dict=feed_dict)
        K.set_learning_phase(1)
        return pred, loss

    def train_batch(self, batch_images, batch_labels):
        # K.set_learning_phase(1)
        self.sess.run(self.init)
        feed_dict = {
            self.images: batch_images,
            self.labels: batch_labels
        }
        loss, _ = self.sess.run(
            [self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def test_batch(self, batch_images):
        pass

    def eval_batch(self):
        pass

    def eval_step(self):
        pass

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
    K.set_learning_phase(1)
    graph = tf.Graph()
    sess = tf.Session()
    config = Config()
    model = Model(config, sess, graph)
    train_data, validation_data, test_data = du.process_data()
    batches = du.generate_train_batches(train_data, config.batch_size)
    batch = du.get_next_batch(batches)
    batch_images, batch_labels = map(list, zip(*batch))
    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)
    batch_images = batch_images.reshape(-1, config.image_size, config.image_size,
                                        config.channels)
    pred, loss = model.predict(batch_images, batch_labels)
    print(loss)

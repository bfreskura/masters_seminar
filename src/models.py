import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class CNN_BILSTM():
    """
    Creates a CNN -> BI-LSTM model architecture
    """

    def __init__(self, config):
        self.learning_rate = config["lr"]
        self.optimizer = config["optimizer"]
        self.timestep = config["timestep"]
        self.word_embd_vec = config["word_vector"]
        self.max_word_size = config["max_word_size"]
        self.char_features = config["char_features"]
        self.cnn_filter = config["filter_dim"]
        self.lstm_hidden = config["lstm_hidden"]
        self.n_classes = config["n_classes"]

        # Architecture
        """
        Word embeddings input of size (batch_size, timestep, word_embed_dim)
        """
        self.word_embedding_input = tf.placeholder(tf.float32,
                                                   (None, self.timestep,
                                                    self.word_embd_vec))

        """
        Character embeddings input of size (batch_size, max_sentence_length
         (a.k.a. timestep), char_embedding_vector_size, max_word_size)
        """
        self.char_inputs = tf.placeholder(tf.float32,
                                          [None,
                                           self.timestep,
                                           self.char_features,
                                           self.max_word_size])

        # POS tags encoded in one-hot fashion (batch_size, num_classes)
        self.labels = tf.placeholder(tf.float32,
                                     (None, self.timestep, self.n_classes))

        # CONVOLUTION on character level
        # Reshape the input so it fits a 2D convolution layer
        net = tf.reshape(self.char_inputs,
                         shape=[-1, self.max_word_size, self.char_features,
                                self.timestep])
        net = tf.layers.conv2d(
            inputs=net,
            filters=self.cnn_filter,
            kernel_size=[3, self.char_features],
            strides=[1, 1],
            padding="SAME",
            activation=tf.nn.relu,
            name="conv1")
        net = tf.layers.max_pooling2d(net,
                                      pool_size=[2, self.char_features],
                                      strides=2, name="pool1")

        net = tf.reshape(net, [-1, self.timestep,
                               self.char_features * self.cnn_filter],
                         name="cnn_flatten")

        # Concat word and char-cnn embeddings
        net = tf.concat([self.word_embedding_input, net], axis=2,
                        name="concat1")

        # BI-LSTM
        # Define weights for the Bi-directional LSTM
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward +
            # backward cells
            'out': tf.Variable(
                tf.random_normal([None, 2 * self.lstm_hidden, self.n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.zeros([None, self.n_classes]))
        }

        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)
        # Get lstm cell output
        net, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                 lstm_bw_cell, net,
                                                 dtype=tf.float32,
                                                 initial_state_bw=tf.random_uniform(
                                                     ([None,
                                                       self.lstm_hidden]),
                                                     minval=-np.sqrt(
                                                         6 / self.lstm_hidden),
                                                     maxval=np.sqrt(
                                                         6 / self.lstm_hidden)),
                                                 initial_state_fw=tf.random_uniform(
                                                     ([None,
                                                       self.lstm_hidden]),
                                                     minval=-np.sqrt(
                                                         6 / self.lstm_hidden),
                                                     maxval=np.sqrt(
                                                         6 / self.lstm_hidden)))

        # Linear activation, using rnn inner loop last output
        pred = tf.matmul(net[-1], weights['out']) + biases['out']

        # Loss and optimizer
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                    labels=self.labels))
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)
        # TODO add CRF layer

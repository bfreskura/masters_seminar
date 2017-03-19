import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class BLSTM_CNN():
    def __init__(self, config):
        self.learning_rate = config["lr"]
        self.optimizer = config["optimizer"]
        self.timestep = config["timestep"]
        self.word_embd_vec = config["word_vector"]
        self.max_word_size = config["max_word_size"]
        self.char_features = config["char_features"]
        self.char_vocab_size = config["char_features"]
        self.cnn_filter = config["filter_dim"]
        self.lstm_hidden = config["lstm_hidden"]
        self.n_classes = config["n_classes"]

        self.char_W = tf.get_variable("char_embed",
                                      [self.char_vocab_size,
                                       self.char_features])

        # Architecture
        self.word_embedding_input = tf.placeholder(
            (None, self.timestep, self.word_embd_vec))

        self.char_inputs = tf.placeholder(tf.int32, [None, self.timestep,
                                                     self.max_word_size])

        self.labels = tf.placeholder((None, self.n_classes))


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
        # CONVOLUTION


        # BI LSTM
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


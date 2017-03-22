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
        self.sess = tf.Session()

        """
        Word embeddings input of size (batch_size, timestep, word_embed_dim)
        """
        self.word_embedding_input = tf.placeholder(tf.float32,
                                                   (None, self.timestep,
                                                    self.word_embd_vec))
        """
        Character embeddings input of size (batch_size, max_sentence_length
         (a.k.a. timestep) * max_word_size)
        """
        self.char_embedding_input = tf.placeholder(tf.int32,
                                                   (None,
                                                    self.timestep * self.max_word_size))
        # POS tags encoded in one-hot fashion (batch_size, num_classes)
        self.labels = tf.placeholder(tf.float32,
                                     (None, self.timestep, self.n_classes))

        char_embed = tf.Variable(
            tf.random_uniform(
                [self.max_word_size * self.timestep,
                 self.char_features],
                minval=-np.sqrt(3 / self.char_features),
                maxval=np.sqrt(3 / self.char_features)),
        )
        net = tf.nn.embedding_lookup(char_embed, self.char_embedding_input)
        net = tf.layers.dropout(net, rate=0.5)

        # CONVOLUTION on character level
        net = tf.layers.conv1d(
            inputs=net,
            filters=self.cnn_filter,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=tf.nn.relu,
            name="conv1")
        net = tf.layers.max_pooling1d(net,
                                      pool_size=2,
                                      strides=2,
                                      name="pool1")
        net = tf.reshape(net, [-1, self.timestep, self.cnn_filter * 10],
                         name="reshape1")

        # Concat word and char-cnn embeddings
        net = tf.concat([self.word_embedding_input, net], axis=2,
                        name="concat1")

        # Apply dropout and prepare input for the LSTM net
        net = tf.layers.dropout(net, rate=0.5)
        net = tf.reshape(net, [-1, self.cnn_filter * 10 + self.word_embd_vec],
                         name="reshape2")
        net = tf.split(net, self.timestep, axis=0, name="split1")

        # BI-LSTM
        # Define weights for the Bi-directional LSTM
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward +
            # backward cells
            'out': tf.Variable(
                tf.random_normal([2 * self.lstm_hidden, self.n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)

        net, _, _ = rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,
                                                 cell_bw=lstm_bw_cell,
                                                 inputs=net,
                                                 dtype=tf.float32)

        # Linear activation, using rnn inner loop on all outputs
        pred = [tf.layers.dropout(tf.matmul(n, weights['out']) + biases['out'],
                                  rate=0.5) for n in net]
        # Softmax probabilites

        # TODO add CRF layer

        pred = tf.reshape(pred, [-1, self.timestep, self.n_classes])
        self.softmax = tf.nn.softmax(pred)

        # Define loss
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                    labels=self.labels))

        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import logging


class CNN_BILSTM_CRF():
    """
    Creates a CNN -> BI-LSTM -> CRF model architecture.
    Reference paper: https://arxiv.org/abs/1603.01354
    """

    def __init__(self, config):
        learning_rate = config["lr"]
        self.optimizer = config["optimizer"]
        self.timestep = config["timestep"]
        self.word_embd_vec = config["word_vector_dim"]
        self.max_word_size = config["max_word_size"]
        self.char_features = config["char_embeddings_dim"]
        self.cnn_filter = config["filter_dim"]
        self.lstm_hidden = config["lstm_hidden"]
        self.n_classes = config["n_classes"]
        self.char_vocab_dim = config["char_vocab_dim"]
        train_examples = config["train_examples"]
        batch_size = config["batch_size"]

        self.sess = tf.Session()
        self.global_step = tf.Variable(0, trainable=False)

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
        self.labels = tf.placeholder(tf.int32,
                                     (None, self.timestep, self.n_classes))

        # Char embedding layer
        char_embed = tf.Variable(
            tf.random_uniform(
                [self.char_vocab_dim, self.char_features],
                minval=-np.sqrt(3 / self.char_features),
                maxval=np.sqrt(3 / self.char_features)),
        )
        net = tf.nn.embedding_lookup(char_embed, self.char_embedding_input)
        net = tf.layers.dropout(net, rate=0.5)

        # 1-D Convolution on a character level
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

        # Concatenate word and char-cnn embeddings
        net = tf.concat([self.word_embedding_input, net], axis=2,
                        name="concat1")

        # Apply dropout and prepare input for the BI-LSTM net
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
                tf.random_uniform([2 * self.lstm_hidden, self.n_classes],
                                  minval=-np.sqrt(6 / (
                                      2 * self.lstm_hidden + self.n_classes)),
                                  maxval=np.sqrt(6 / (
                                      2 * self.lstm_hidden + self.n_classes)))
            )
        }
        biases = {
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }

        # Forward and backward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)

        net, _, _ = rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,
                                                 cell_bw=lstm_bw_cell,
                                                 inputs=net,
                                                 dtype=tf.float32)

        # Linear activation, using rnn inner loop on all outputs
        pred = [
            tf.layers.dropout(tf.matmul(n, weights['out']) + biases['out'], 0.5)
            for n in net]
        self.logits = tf.reshape(pred, [-1, self.timestep, self.n_classes])

        # CRF Layer
        sequence_lengths = np.full(batch_size, self.timestep - 1,
                                   dtype=np.int32)
        sequence_lengths_t = tf.constant(sequence_lengths)

        crf_logits, self.trans_params = tf.contrib.crf.crf_log_likelihood(
            self.logits,
            tf.cast(tf.argmax(self.labels, axis=2), tf.int32),
            sequence_lengths_t)

        # Loss and learning rate
        self.loss = tf.reduce_mean(-crf_logits)
        self.lr = tf.train.exponential_decay(learning_rate,
                                             global_step=self.global_step,
                                             decay_steps=train_examples // batch_size,
                                             decay_rate=0.95)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(self.loss,
                                            global_step=self.global_step)

    def load_model(self, model_path):
        """
        Restores the model from the checkpoint file
        :param model_path:
        :return:
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        logging.info("Model restored from " + model_path)

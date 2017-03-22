"""
Model training and evaluation
"""
import IPython
import utils

import tensorflow as tf


def train(train_word,
          valid_word,
          train_label,
          valid_label,
          model,
          batch_size=32,
          num_epochs=5):
    # Init variables
    model.sess.run(tf.global_variables_initializer())

    num_batches = train_word.shape[0] // batch_size

    for epoch in range(num_epochs):
        # Shuffle training data in each epoch
        train_word, train_label = utils.shuffle_data(train_word, train_label)
        for b in range(num_batches):
            word = train_word[b * batch_size:(b + 1) * batch_size]
            label = train_label[b * batch_size:(b + 1) * batch_size]
            loss, _, pred = model.sess.run(
                [model.loss, model.train_op, model.softmax],
                {model.word_embedding_input: word,
                 model.labels: label})

            if b % 10 == 0:
                print("Batch Loss {:.4f}".format(loss))

        print("Finished epoch {}".format(epoch + 1))


def eval():
    pass

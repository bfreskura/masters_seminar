"""
Model training and evaluation
"""
import tensorflow as tf
import IPython
import numpy as np

import utils


def train(train_word,
          valid_word,
          train_chr,
          valid_chr,
          train_label,
          valid_label,
          model,
          batch_size=4,
          num_epochs=20):
    # Init variables
    model.sess.run(tf.global_variables_initializer())

    num_batches = train_word.shape[0] // batch_size

    for epoch in range(num_epochs):
        # Shuffle training data in each epoch
        train_chr, train_word, train_label = utils.shuffle_data(train_chr,
                                                                train_word,
                                                                train_label)
        for b in range(num_batches):
            chr = train_chr[b * batch_size:(b + 1) * batch_size]
            word = train_word[b * batch_size:(b + 1) * batch_size]
            label = train_label[b * batch_size:(b + 1) * batch_size]
            loss, _, pred = model.sess.run(
                [model.loss, model.train_op, model.softmax],
                {model.char_embedding_input: chr,
                 model.word_embedding_input: word,
                 model.labels: label})

            if b % 20 == 0:
                print("Iteration {}/{}, Batch Loss {:.4f}".format(
                    b * batch_size, num_batches * batch_size, loss))

                eval(model, valid_chr, valid_word, valid_label)
        print("Finished epoch {}\n".format(epoch + 1))


def eval(model, chr, word, label):
    pred = model.sess.run(
        [model.softmax],
        {model.char_embedding_input: chr,
         model.word_embedding_input: word,
         model.labels: label})
    pred = np.argmax(pred[0], axis=2)
    true = np.argmax(label, axis=2)
    print("Validation Accuracy",
          np.sum(pred == true) / (pred.shape[0] * pred.shape[1]))

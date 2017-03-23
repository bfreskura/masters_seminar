"""
Model training and evaluation
"""
import numpy as np
import tensorflow as tf
import constants
import os

import utils


def train(train_word,
          valid_word,
          train_chr,
          valid_chr,
          train_label,
          valid_label,
          model,
          batch_size=4,
          num_epochs=20,
          domain="treebank_wjs",
          model_save_dir=constants.TF_WEIGHTS):
    """

    :param train_word:
    :param valid_word:
    :param train_chr:
    :param valid_chr:
    :param train_label:
    :param valid_label:
    :param model:
    :param batch_size:
    :param num_epochs:
    :param model_save_dir:
    :return:
    """
    # Init variables
    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

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
            loss, _, pred, lr = model.sess.run(
                [model.loss, model.train_op, model.softmax, model.lr],
                {model.char_embedding_input: chr,
                 model.word_embedding_input: word,
                 model.labels: label})

            if b % 40 == 0:
                print("Iteration {}/{}, Batch Loss {:.4f}, LR: {:.4f}".format(
                    b * batch_size, num_batches * batch_size, loss, lr))
                print(model.sess.run(model.global_step))

        eval(model, valid_chr, valid_word, valid_label)
        print("Finished epoch {}\n".format(epoch + 1))

        if epoch % 30 == 0:
            # Save model every n epochs
            saver.save(model.sess,
                       os.path.join(model_save_dir,
                                    domain + "_cnn_bilstm_crf.ckpt"),
                       global_step=model.global_step)


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

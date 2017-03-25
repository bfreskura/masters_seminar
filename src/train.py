"""
Model training and evaluation
"""
import os
import IPython
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, \
    precision_score

import constants
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

    :param file_log:
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

    # TODO add file logging
    for epoch in range(num_epochs):
        # Shuffle training data in each epoch
        train_chr, train_word, train_label = utils.shuffle_data(train_chr,
                                                                train_word,
                                                                train_label)
        for b in range(num_batches):
            chr = train_chr[b * batch_size:(b + 1) * batch_size]
            word = train_word[b * batch_size:(b + 1) * batch_size]
            label = train_label[b * batch_size:(b + 1) * batch_size]
            loss, _, lr = model.sess.run(
                [model.loss, model.train_op, model.lr],
                {model.char_embedding_input: chr,
                 model.word_embedding_input: word,
                 model.labels: label})

            if (b + 1) % 5 == 0:
                print("Iteration {}/{}, Batch Loss {:.4f}, LR: {:.4f}".format(
                    b * batch_size, num_batches * batch_size, loss, lr))

        eval(model, valid_chr, valid_word, valid_label,
             batch_size=batch_size)
        print("Finished epoch {}\n".format(epoch + 1))

        if (epoch + 1) % 30 == 0:
            # Save model every n epochs
            path = saver.save(model.sess,
                              os.path.join(model_save_dir,
                                           domain + "_cnn_bilstm_crf.ckpt"),
                              global_step=model.global_step)
            print("Model saved at", path)


def eval(model, char, word, label, batch_size=256):
    """
    Evaluates the model using standard evaluation metrics
    :param model: Trained Model
    :param char: List of char embeddings
    :param word: List of word embeddings
    :param label: List of labels
    :param batch_size: Batch size
    :return:
    """
    print("Evaluating on the validation set...")
    num_batches = char.shape[0] // batch_size
    acc, prec, rec, f1 = 0, 0, 0, 0
    for b in range(num_batches):
        chr_b = char[b * batch_size:(b + 1) * batch_size]
        word_b = word[b * batch_size:(b + 1) * batch_size]
        label_b = label[b * batch_size:(b + 1) * batch_size]
        loss, unary_scores, trans_params = model.sess.run(
            [model.loss, model.logits, model.trans_params],
            {model.char_embedding_input: chr_b,
             model.word_embedding_input: word_b,
             model.labels: label_b})

        # Get CRF output scores
        true_batch = []
        for tf_unary_scores_ in unary_scores:
            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, trans_params)
            true_batch.append(viterbi_sequence)

        # Update metric
        a, p, r, f = calc_metric(np.array(true_batch),
                                 np.argmax(label_b, axis=2))
        acc += a
        prec += p
        rec += r
        f1 += f

    print("Accuracy {:.3f}%".format(acc / num_batches * 100))
    print("Macro Precision {:.3f}%".format(prec / num_batches * 100))
    print("Macro Recall {:.3f}%".format(rec / num_batches * 100))
    print("Macro F1 {:.3f}%\n".format(f1 / num_batches * 100))


def calc_metric(y_trues, y_preds):
    """
    Calculates accuracy, macro preicision, recall and f1 scores
    :param y_true: Batch of true labels
    :param y_pred: Batch of predicted labels
    :return:
    """
    assert len(y_trues) == len(y_preds)

    acc, prec, rec, f1, dim = 0, 0, 0, 0, y_trues.shape[0]
    for y_true, y_pred in zip(y_trues, y_preds):
        acc += accuracy_score(y_true, y_pred)
        prec += precision_score(y_true, y_pred, average="macro")
        rec += recall_score(y_true, y_pred, average="macro")
        f1 += f1_score(y_true, y_pred, average="macro")

    return acc / dim, prec / dim, rec / dim, f1 / dim

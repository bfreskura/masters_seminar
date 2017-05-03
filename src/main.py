import datetime
import logging
import os
import tensorflow as tf

import numpy as np

import constants
import data_loader
import models
import train
import utils


def main(config, download_resources=False,
         process_data=False, test_size=0.4,
         model_train=False, model_path=None):
    """

    :param config:
    :param download_resources:
    :param process_data:
    :param test_size:
    :param model_train:
    :param model_path:
    :return:
    """
    if download_resources:
        utils.download_data()

    # Get data
    if config['domain'] == "NER_CONLL":
        train_chr, valid_chr, test_chr, train_word, valid_word, test_word, train_label, \
        valid_label, test_label, chr_id_mappings, = data_loader.prepare_ner_data(
            process_data, test_size)
    else:
        train_chr, valid_chr, test_chr, train_word, valid_word, test_word, train_label, \
        valid_label, test_label, chr_id_mappings, = data_loader.prepare_wjs_data(
            process_data, test_size)

    # Update config
    config['n_classes'] = train_label.shape[2]
    config['char_vocab_dim'] = len(chr_id_mappings) + 1
    config['train_examples'] = train_chr.shape[0]
    config['validation_examples'] = valid_chr.shape[0]
    config['test_examples'] = test_chr.shape[0]

    logging.info("CONFIG:")
    logging.info("\n".join([k + ": " + str(v) for k, v in config.items()]))

    model = models.CNN_BILSTM_CRF(config)

    if model_train:
        train.train(train_word=train_word,
                    valid_word=valid_word,
                    train_chr=train_chr,
                    valid_chr=valid_chr,
                    train_label=train_label,
                    valid_label=valid_label,
                    num_epochs=config['train_epochs'],
                    model=model,
                    batch_size=config['batch_size'],
                    config=config)
        # Evaluate at the end
        logging.info("Evaluating at the TEST set")
        train.eval(model, test_chr, test_word, test_label, config['batch_size'])

    else:
        if model_path:
            saver = tf.train.Saver()
            saver.restore(model.sess, model_path)
            # Test the model on the test set
            logging.info("Evaluating at the TEST set")
            train.eval(model, test_chr, test_word, test_label,
                       config['batch_size'])
        else:
            print("No trained models exist! You have to train the model first.")


if __name__ == "__main__":
    # For results consistency
    seed = 1337
    np.random.seed(seed)
    config = utils.read_config(os.path.join(constants.CONFIGS, "wsj_model.ini"))

    # Setup logging
    utils.dir_creator([constants.LOGS, constants.TF_WEIGHTS])
    log_name = config['domain'] + "_" + str(
        datetime.datetime.now().strftime("%d_%m_%Y_%H:%M")) + ".log"
    log_file = os.path.join(constants.LOGS, log_name)
    print("Logging to", log_file)
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.DEBUG, datefmt='%d/%m/%Y %I:%M:%S %p')

    logging.info("Numpy random seed set to " + str(seed))
    main(process_data=True, config=config, model_train=True)

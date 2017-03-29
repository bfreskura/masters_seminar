import os

import numpy as np
import logging
import constants
import data_loader
import input_process
import models
import train
import utils
import datetime


def prepare_ner_data(process_data, test_size):
    pass


def prepare_wjs_data(process_data, test_size):
    """

    :param process_data:
    :param test_size:
    :return:
    """
    # Setup paths
    path_prefix = os.path.join(constants.WJS_DATA,
                               "wjs_treebank_timest" + str(
                                   constants.TIMESTEP) + "_")
    word_embeddings_path = path_prefix + "word_embeddings.pkl"
    char_id_mappings_path = path_prefix + "char_id_mappings.pkl"
    char_embeddings_path = path_prefix + "char_embeddings.pkl"
    labels_path = path_prefix + "onehot_labels.pkl"

    if process_data:
        wjs_data = data_loader.parse_WJS(constants.WJS_DATA_DIR)
        input_process.embed_words(wjs_data, export_file=word_embeddings_path)
        input_process.create_char_mappings(wjs_data,
                                           export_file=char_embeddings_path,
                                           mappings_export_file=char_id_mappings_path)
        input_process.encode_labels(wjs_data, export_file=labels_path)

    # Load data from disk
    treebank = utils.load_pickle(word_embeddings_path)
    pos_tags = utils.load_pickle(labels_path)
    chr_embds = utils.load_pickle(char_embeddings_path)
    chr_id_mappings = utils.load_pickle(char_id_mappings_path)

    # Shuffle
    chr_embds, treebank, pos_tags = utils.shuffle_data(chr_embds, treebank,
                                                       pos_tags)

    # Split
    train_chr, valid_chr, train_word, valid_word, train_label, \
    valid_label = utils.split_data(chr_embds, treebank, pos_tags,
                                   test_size=test_size)

    return chr_id_mappings, train_chr, valid_chr, train_word, valid_word, \
           train_label, valid_label, chr_id_mappings


def main(download_and_process_data=False, process_data=False, test_size=0.3,
         batch_size=128, learning_rate=1e-2):
    if download_and_process_data:
        utils.download_data()

    chr_id_mappings, train_chr, valid_chr, train_word, valid_word, train_label, \
    valid_label, chr_id_mappings = prepare_wjs_data(process_data, test_size)

    # Net config
    config = {
        "lr": learning_rate,
        "optimizer": "Adam",
        "timestep": constants.TIMESTEP,
        "word_vector_dim": 100,
        "max_word_size": constants.MAX_WORD_SIZE,
        "char_embeddings_dim": constants.CHAR_EMBEDDINGS_FEATURE,
        "filter_dim": 30,
        "lstm_hidden": 200,
        "n_classes": train_label.shape[2],
        "batch_size": batch_size,
        "train_examples": train_chr.shape[0],
        "char_vocab_dim": len(chr_id_mappings) + 1
    }
    logging.info(" ".join(["CONFIG:", str(config)]))
    model = models.CNN_BILSTM_CRF(config)

    train.train(train_word=train_word,
                valid_word=valid_word,
                train_chr=train_chr,
                valid_chr=valid_chr,
                train_label=train_label,
                valid_label=valid_label,
                num_epochs=50,
                model=model,
                batch_size=config['batch_size'])


if __name__ == "__main__":
    # For results consistency
    seed = 1337
    np.random.seed(seed)

    # Setup logging
    utils.dir_creator([constants.LOGS])
    log_name = str(
        datetime.datetime.now().strftime("%d_%m_%Y_%H:%M")) + ".log"
    log_file = os.path.join(constants.LOGS, log_name)
    print("Logging to", log_file)
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.DEBUG, datefmt='%d/%m/%Y %I:%M:%S %p')
    logging.info("Numpy random seed set to " + str(seed))
    main()

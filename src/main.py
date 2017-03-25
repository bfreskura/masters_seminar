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


def main(download_and_process_data=False, process_data=False, test_size=0.3,
         batch_size=128, learning_rate=1e-2):
    if download_and_process_data:
        utils.download_data()

    if process_data:
        wjs_data = data_loader.parse_WJS(constants.WJS_DATA_DIR)
        input_process.embed_words(wjs_data)
        input_process.create_char_mappings(wjs_data,
                                           export_dir=constants.WJS_DATA)
        input_process.encode_labels(wjs_data)

    treebank = utils.load_pickle(os.path.join(constants.WJS_DATA,
                                              "wjs_treebank_glove_100_t" + str(
                                                  constants.TIMESTEP) + ".pkl"))
    pos_tags = utils.load_pickle(os.path.join(constants.WJS_DATA,
                                              "wjs_pos_one_hot_" + str(
                                                  constants.TIMESTEP) + ".pkl"))
    chr_embds = utils.load_pickle(os.path.join(constants.WJS_DATA,
                                               "treebank_wjs_char_mappings_" + str(
                                                   constants.TIMESTEP) + ".pkl"))
    chr_id_mappings = utils.load_pickle(
        os.path.join(constants.WJS_DATA, "treebank_wjs_chr_id_mappings.pkl"))

    # Shuffle
    chr_embds, treebank, pos_tags = utils.shuffle_data(chr_embds, treebank,
                                                       pos_tags)

    # Split
    train_chr, valid_chr, train_word, valid_word, train_label, valid_label = utils.split_data(
        chr_embds, treebank, pos_tags, test_size=test_size)

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
        "n_classes": pos_tags.shape[2],
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
    np.random.seed(1337)

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
    main()

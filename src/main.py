import datetime
import logging
import os

import numpy as np

import constants
import data_loader
import models
import train
import utils


def main(download_and_process_data=False, process_data=False, test_size=0.3,
         batch_size=128, learning_rate=1e-2):
    if download_and_process_data:
        utils.download_data()

    # chr_id_mappings, train_chr, valid_chr, train_word, valid_word, train_label, \
    # valid_label, chr_id_mappings = data_loader.prepare_wjs_data(process_data,
    #                                                             test_size)

    chr_id_mappings, train_chr, valid_chr, train_word, valid_word, train_label, \
    valid_label, chr_id_mappings = data_loader.prepare_ner_data(process_data,
                                                                test_size)

    # # Net config
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
                domain="NER",
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

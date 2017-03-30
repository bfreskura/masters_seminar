import datetime
import logging
import os

import numpy as np

import constants
import data_loader
import models
import train
import utils


def main(download_resources=False,
         process_data=False, test_size=0.3):
    if download_resources:
        utils.download_data()

    # Load Net config
    config = utils.read_config(os.path.join(constants.CONFIGS, "ner_model.ini"))
    if config['domain'] == "NER":
        chr_id_mappings, train_chr, valid_chr, train_word, valid_word, train_label, \
        valid_label, chr_id_mappings = data_loader.prepare_ner_data(
            process_data,
            test_size)
    else:
        chr_id_mappings, train_chr, valid_chr, train_word, valid_word, train_label, \
        valid_label, chr_id_mappings = data_loader.prepare_wjs_data(
            process_data,
            test_size)

    # Update config
    config['n_classes'] = train_label.shape[2]
    config['char_vocab_dim'] = len(chr_id_mappings) + 1
    config['train_examples'] = train_chr.shape[0]

    logging.info("CONFIG:")
    logging.info("\n".join([k + ": " + str(v) for k, v in config.items()]))

    model = models.CNN_BILSTM_CRF(config)

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
    main(process_data=False)

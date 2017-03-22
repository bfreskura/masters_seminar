import os

import IPython
import numpy as np

import constants
import data_loader
import utils
import input_process
import train
import models


def main():
    # wjs_data = data_loader.parse_WJS(constants.WJS_DATA_DIR)
    # input_process.embed_words(wjs_data)
    # input_process.embed_chars(wjs_data)
    # input_process.create_char_mappings(wjs_data)

    treebank = utils.load_pickle(os.path.join(constants.WJS_DATA,
                                              "wjs_treebank_glove_100_t" + str(
                                                  constants.TIMESTEP) + ".pkl"))
    pos_tags = utils.load_pickle(os.path.join(constants.WJS_DATA,
                                              "wjs_treebank_pos_tags_one_hot_" + str(
                                                  constants.MAX_WORD_SIZE) + ".pkl"))

    chr_embds = utils.load_pickle(os.path.join(constants.WJS_DATA, "wjs_char_id.pkl"))

    # # Shuffle
    treebank, pos_tags = utils.shuffle_data(treebank, pos_tags)

    train_word, valid_word = treebank[:3000], treebank[3000:]
    train_label, valid_label = pos_tags[:3000], pos_tags[3000:]

    config = {
        "lr": 1.5e-4,
        "optimizer": "Adam",
        "timestep": constants.TIMESTEP,
        "word_vector": 100,
        "max_word_size": constants.MAX_WORD_SIZE,
        "char_features": constants.CHAR_EMBEDDINGS_FEATURE,
        "filter_dim": 30,
        "lstm_hidden": 200,
        "n_classes": pos_tags.shape[2],
        "batch_size": 32,
    }

    model = models.CNN_BILSTM(config)

    train.train(train_word=train_word,
                valid_word=valid_word,
                train_label=train_label,
                valid_label=valid_label,
                model=model,
                batch_size=config['batch_size'])


    if __name__ == "__main__":
    # For results consistency
        np.random.seed(1337)
    main()

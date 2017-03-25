import os

import numpy as np

import constants
import models
import train
import utils
import input_process
import data_loader
import IPython


def split_data(chr_embds, treebank, pos_tags, test_size=0.3):
    ts = int((1 - test_size) * chr_embds.shape[0])

    train_chr, valid_chr = chr_embds[:ts], chr_embds[ts:]
    train_word, valid_word = treebank[:ts], treebank[ts:]
    train_label, valid_label = pos_tags[:ts], pos_tags[ts:]

    return train_chr, valid_chr, train_word, valid_word, train_label, valid_label


def main():
    # wjs_data = data_loader.parse_WJS(constants.WJS_DATA_DIR)
    # input_process.embed_words(wjs_data)
    # input_process.create_char_mappings(wjs_data, export_dir=constants.WJS_DATA)
    # input_process.encode_labels(wjs_data)

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
    train_chr, valid_chr, train_word, valid_word, train_label, valid_label = split_data(
        chr_embds, treebank, pos_tags)

    config = {
        "lr": 1e-2,
        "optimizer": "Adam",
        "timestep": constants.TIMESTEP,
        "word_vector": 100,
        "max_word_size": constants.MAX_WORD_SIZE,
        "char_features": constants.CHAR_EMBEDDINGS_FEATURE,
        "filter_dim": 30,
        "lstm_hidden": 200,
        "n_classes": pos_tags.shape[2],
        "batch_size": 64,
        "train_examples": train_chr.shape[0],
        "char_vocab_dim": len(chr_id_mappings) + 1
    }
    model = models.CNN_BILSTM_CRF(config)

    print("CONFIG:", config)
    train.train(train_word=train_word,
                valid_word=valid_word,
                train_chr=train_chr,
                valid_chr=valid_chr,
                train_label=train_label,
                valid_label=valid_label,
                num_epochs=100,
                model=model,
                batch_size=config['batch_size'])


if __name__ == "__main__":
    # For results consistency
    np.random.seed(1337)
    main()

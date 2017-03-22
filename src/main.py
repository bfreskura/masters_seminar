import os

import IPython
import numpy as np

import constants
import data_loader
import utils
import input_process


def main():
    wjs_data = data_loader.parse_WJS(constants.WJS_DATA_DIR)
    # # input_process.embed_words(wjs_data)
    input_process.embed_chars(wjs_data)

    # chr_embd = utils.load_pickle(
    #     os.path.join(constants.WJS_DATA, "wjs_treebank_char_embedding20.pkl"))


    treebank = utils.load_pickle(os.path.join(constants.WJS_DATA,
                                              "wjs_treebank_glove_100_t" + str(
                                                  constants.TIMESTEP) + ".pkl"))


if __name__ == "__main__":
    # For results consistency
    np.random.seed(1337)
    main()

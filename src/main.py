import data_loader
import os
import constants
import input_process
import utils


def main():
    # wjs_data = data_loader.parse_WJS(constants.WJS_DATA_DIR)
    # input_process.embed_words(wjs_data)

    treebank = utils.load_pickle(os.path.join(constants.WJS_DATA,
                                              "wjs_treebank_glove_100_t" + str(
                                                  constants.TIMESTEP) + ".pkl"))


if __name__ == "__main__":
    main()

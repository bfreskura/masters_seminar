import os

import numpy as np
import IPython

import constants
import utils


def _embed_glove(data, glove_file):
    """
    Creates word embedding using the Glove 100 vectors.
    :param data: List of sentences where each sentence consists of (token, pos_tag)
    pairs
    :return: List of sentences which contain (token, tag, embedding) triplets
    """
    embed_dict = dict()

    # Load glove hashmap
    print("Loading Glove vectors, may take a while")
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            token = split[0]
            vec = np.array([np.float(x) for x in split[1:]])
            embed_dict[token] = vec
    print("Glove vectors loaded.")

    embed_data = []
    for sent in data:
        new_sent = []
        for token, tag in sent:
            try:
                new_sent.append((token, tag, embed_dict[token]))
            except:
                print("Word", token, "does not exist in the glove vector")
        embed_data.append(new_sent)
    return embed_data


def embed_words(data, embedding="glove"):
    """
    Create embedded dataset and exports it to a pickle file
    :param embedding: Embedding type
    :return:
    """
    if embedding.lower() == "glove":
        embeddings = _embed_glove(data,
                                  os.path.join(constants.RESOURCES, "glove_100",
                                               "glove.6B.100d.txt"))
        utils.export_pickle(constants.WJS_DATA, "wjs_treebank_glove_100", embeddings)
    elif embedding.lower() == "word2vec":
        pass
    else:
        raise NotImplementedError(
            "Embedding not supported. Supported embeddings"
            "include `glove`, `word2vec`")

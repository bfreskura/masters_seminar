import os

import IPython
import numpy as np

import constants
import utils


def _embed_glove(data, glove_file, timestep):
    """
    Creates word embedding using the Glove 100 vectors.
    :param data: List of sentences where each sentence consists of (token, pos_tag)
    pairs
    :return: List of sentences which contain (token, tag, embedding) triplets
    """
    embed_dict = dict()
    glove_vec_size = 100

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

        # Adjust sentence length so it fits into a LSTM net
        if len(new_sent) > timestep:
            new_sent = new_sent[:timestep]
            embed_data.append(new_sent)
        elif len(new_sent) < timestep:
            for _ in range(timestep - len(new_sent)):
                new_sent.append(
                    ("NULL", "NULL", np.zeros(glove_vec_size, dtype=np.float)))
            embed_data.append(new_sent)
        else:
            embed_data.append(new_sent)

        assert len(new_sent) == timestep, "Mismatch in timestep size"

    return embed_data


def _embed_w2vec(data, export_file):
    raise NotImplementedError


def _construct_mappings(chars):
    """
    Creates id -> char and char -> id mappings
    :param chars: List of characters in a corpus
    :return:
    """
    char_set = set(chars)
    id_chr = {id: chr for id, chr in zip(range(len(char_set)), char_set)}
    chr_id = {chr: id for id, chr in zip(range(len(char_set)), char_set)}
    return id_chr, chr_id


def _char_to_onehot(chr, chr_id_mapping):
    """
    Convert character to a one hot representation
    :param chr: Character
    :param chr_id_mapping: character to id map
    :return:
    """
    vector = np.zeros(len(chr_id_mapping), dtype=np.uint8)
    vector[chr_id_mapping[chr]] = 1
    return vector


def embed_chars(data, max_word_size=20):
    """

    :param data:
    :return:
    """
    chars = []
    # Create mappings
    for sent in data:
        for token, _ in sent:
            chars.extend(token)
    id_chr, chr_id = _construct_mappings(chars)

    # Convert to one hot
    new_sents = []
    for sent in data:
        sentence = []
        for token, tag in sent:
            char_embeddings = np.zeros((len(id_chr), max_word_size))
            chr_vecs = np.array([_char_to_onehot(c, chr_id) for c in token]).T

            # Trim if the max word size is exceeded
            if chr_vecs.shape[1] > max_word_size:
                char_embeddings = chr_vecs[:, :max_word_size]

            elif chr_vecs.shape[1] < max_word_size:
                pad_size = (max_word_size - chr_vecs.shape[1]) // 2
                for i in range(chr_vecs.shape[1]):
                    char_embeddings[:, i + pad_size] = chr_vecs[:, i]

            else:
                char_embeddings = chr_vecs

            assert char_embeddings.shape[0] == len(id_chr)
            assert char_embeddings.shape[1] == max_word_size
            sentence.append(np.array(char_embeddings))

        new_sents.append(np.array(sentence))

    utils.export_pickle(constants.WJS_DATA,
                        "wjs_treebank_char_embedding" + str(max_word_size),
                        np.array(new_sents))


def embed_words(data, embedding="glove", timestep=constants.TIMESTEP):
    """
    Create embedded dataset and exports it to a pickle file
    :param timestep: Maximum sentence length after embedding
    :param embedding: Embedding type
    :return:
    """
    if embedding.lower() == "glove":
        word_embeddings = _embed_glove(data,
                                       os.path.join(constants.RESOURCES,
                                                    "glove_100",
                                                    "glove.6B.100d.txt"),
                                       timestep=timestep)
        utils.export_pickle(constants.WJS_DATA,
                            "wjs_treebank_glove_100_t" + str(timestep),
                            word_embeddings)
    elif embedding.lower() == "word2vec":
        pass
    else:
        raise NotImplementedError(
            "Embedding not supported. Supported embeddings"
            "include `glove`, `word2vec`")

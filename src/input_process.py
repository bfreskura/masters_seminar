import os

import IPython
import numpy as np

import constants
import utils


def _embed_glove(data, glove_file, timestep, glove_vec_size=100):
    """
    Creates word embedding using the Glove vectors.
    :param data: List of sentences where each sentence consists of (token, pos_tag)
    pairs
    :return: List of sentences which contain (timestep, vec_size) arrays
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
                new_sent.append(embed_dict[token])
            except:
                new_sent.append(np.zeros(glove_vec_size))
                print("Word", token, "does not exist in the glove vector")

        # Adjust sentence length so it fits into a LSTM net
        if len(new_sent) > timestep:
            new_sent = new_sent[:timestep]
            embed_data.append(new_sent)
        elif len(new_sent) < timestep:
            for _ in range(timestep - len(new_sent)):
                new_sent.append(
                    np.zeros(glove_vec_size, dtype=np.float))
            embed_data.append(new_sent)
        else:
            embed_data.append(new_sent)

    embed_data = np.array(embed_data)
    assert embed_data.shape[0] == len(data)
    assert embed_data.shape[1] == timestep
    assert embed_data.shape[2] == glove_vec_size

    return embed_data


def _embed_w2vec(data, export_file):
    raise NotImplementedError


def _construct_mappings(data):
    """
    Creates item -> id mappings
    :param data: List of all items in a corpus
    :return:
    """
    unique_set = set(data)
    return {chr: id for id, chr in
            zip(range(1, len(unique_set) + 1), unique_set)}


def _item_to_onehot(item, item_mappings):
    """
    Convert character to a one hot representation
    :param item: Item
    :param item_mappings: Item mapping hash-map
    :return:
    """
    vector = np.zeros(len(item_mappings), dtype=np.uint8)
    vector[item_mappings[item] - 1] = 1
    return vector


def create_char_mappings(data,
                         max_word_size=constants.MAX_WORD_SIZE,
                         timestep=constants.TIMESTEP,
                         export_dir=None):
    """
    Creates characters mappings in the following manner:
    1) Map each character to its id (using a hash map)
    2) Split the sentence into words
    3) For each word, map each char to its ID, and pad for the max_word_size
    4) Do this for every word in the sentence
    5) Join vectors for one sentence to create a (1, timestep*max_word_size) vector
    6) Repeat 2) - 4) for every sentence
    7) Join everything together and to obtain a
     (num_sentences, timestep * max_word_size) matrix

    Exports data to a pickle file

    :param data: Dataset
    :param max_word_size: Maximum word length
    :param timestep: Maximum Sentence length
    :return: (7), chr->id hashmap) tuple
    """

    # Create char -> id hash map
    chars = []
    for sent in data:
        for token, tag in sent:
            chars.extend(token)
    chr_rvec = _construct_mappings(chars)

    sentences = []
    for sent in data:
        new_sent = np.zeros(timestep * max_word_size, dtype=np.uint8)

        new_sent_temp = []
        for token, _ in sent:
            char_embeddings = np.zeros(max_word_size, dtype=np.uint8)
            char_mapped = np.array([chr_rvec[c] for c in token])

            if len(char_mapped) > max_word_size:
                char_embeddings = char_mapped[:max_word_size]
            elif len(char_mapped) < max_word_size:
                pad_size = (max_word_size - len(char_mapped)) // 2
                for i in range(len(char_mapped)):
                    char_embeddings[i + pad_size] = char_mapped[i]
            else:
                char_embeddings = char_mapped
            new_sent_temp.extend(char_embeddings)

        new_sent_temp = np.array(new_sent_temp)
        if len(new_sent_temp) > timestep * max_word_size:
            new_sent = new_sent_temp[:len(new_sent)]

        elif len(new_sent_temp) < timestep * max_word_size:
            new_sent[:len(new_sent_temp)] = new_sent_temp
        else:
            new_sent = new_sent_temp

        sentences.append(new_sent)

    sentences = np.array(sentences, dtype=np.int32)
    assert sentences.shape[0] == len(data)
    assert sentences.shape[1] == timestep * max_word_size

    # Export to pickle
    if export_dir:
        utils.export_pickle(export_dir,
                            "treebank_wjs_char_mappings_" + str(timestep),
                            np.array(sentences))

    # Export mappings
    utils.export_pickle(export_dir,
                        "treebank_wjs_chr_id_mappings", chr_rvec)
    return np.array(sentences)


def encode_labels(data, timestep=constants.TIMESTEP):
    """
    TODO
    :param data:
    :param timestep:
    :return:
    """
    tags = []
    for sent in data:
        for token, tag in sent:
            tags.append(tag)
    tags.append(constants.WJS_NONE_TAG)
    tag_id = _construct_mappings(tags)

    encoded_tags = []
    for sent in data:
        new_enc = np.zeros((timestep, len(tag_id)), dtype=np.uint8)
        encoded = np.array([_item_to_onehot(tag, tag_id) for _, tag in sent],
                           dtype=np.uint8)

        if len(encoded) > timestep:
            new_enc = encoded[:timestep]
        elif len(encoded) < timestep:
            new_enc[:len(encoded)] = encoded[:]
        else:
            new_enc = encoded
        encoded_tags.append(new_enc)

    encoded_tags = np.array(encoded_tags)
    assert encoded_tags.shape[0] == len(data)
    assert encoded_tags.shape[1] == timestep
    assert encoded_tags.shape[2] == len(tag_id)

    utils.export_pickle(constants.WJS_DATA,
                        "wjs_pos_one_hot_" + str(timestep), encoded_tags)
    return encoded_tags


def embed_words(data, embedding="glove", timestep=constants.TIMESTEP):
    """
    Create word embedded dataset and exports it to a pickle file
    :param timestep: Maximum sentence length after embedding
    :param embedding: Embedding type
    :return:
    """
    if embedding.lower() == "glove":
        word_embeddings = _embed_glove(data,
                                       os.path.join(constants.RESOURCES,
                                                    "glove",
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

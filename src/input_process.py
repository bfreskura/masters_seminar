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
                new_sent.append(embed_dict[token])
            except:
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

        assert len(new_sent) == timestep, "Mismatch in timestep size"

    return np.array(embed_data)


def _embed_w2vec(data, export_file):
    raise NotImplementedError


def _construct_mappings(data):
    """
    Creates item -> id mappings
    :param data: List of all items in a corpus
    :return:
    """
    unique_set = set(data)
    return {chr: id for id, chr in zip(range(len(unique_set)), unique_set)}


def _construct_random_mappings(chars, char_embed_dim=30):
    """
    Creates a [char -> random vector] hashmap
    :param chars: List of all characters in a corpus
    :return: Char -> random vector hashmap
    """
    # Get unique chars
    char_set = set(chars)

    # Create randomly initialized character embeddings of shape
    # (unique_chrs, word_length). Using initialization from the paper.
    chr_vecs = np.random.uniform(-np.sqrt(3 / char_embed_dim),
                                 np.sqrt(3 / char_embed_dim),
                                 size=(len(char_set), char_embed_dim))
    chr_id = {chr: id for id, chr in zip(chr_vecs, char_set)}
    return chr_id


def _item_to_onehot(item, item_mappings):
    """
    Convert character to a one hot representation
    :param item: Item
    :param item_mappings: Item mapping hash-map
    :return:
    """
    vector = np.zeros(len(item_mappings), dtype=np.uint8)
    vector[item_mappings[item]] = 1
    return vector


def embed_chars(data,
                max_word_size=constants.MAX_WORD_SIZE,
                timestep=constants.TIMESTEP,
                char_embed_dim=constants.CHAR_EMBEDDINGS_FEATURE,
                dataset="wjs"):
    """
    Embed dataset on a character level. Each character is represented as a
    random char_embed_dim dimensional vector. Exports embedding to a pickle file.
    Final list item structure is the following:
    (sentence_length, char_embed_dim, max_word_size)

    :param timestep: Timestep dimension. Represents the maximum supported
     sentence length.
    :param dataset: Dataset type
    :param char_embed_dim: Character embeddings vector dimension. 30 is set
    according to the paper.
    :param max_word_size: Words with length less than this are padded with zeros
    and words with greater length are trimmed to max_word_size.
    :param data: List of sentences where each sentence consists of (token, pos_tag)
    pairs
    :return:
    """
    chars = []
    taggings = []
    for sent in data:
        for token, tag in sent:
            chars.extend(token)
            taggings.append(tag)

    # Append the special tag
    taggings.append(constants.WJS_NONE_TAG)

    chr_rvec = _construct_random_mappings(chars)
    tag_ids = _construct_mappings(taggings)

    # Export character mappings
    utils.export_pickle(constants.RESOURCES, "chr_vector_mappings_" + dataset,
                        chr_rvec)

    new_sents = []
    new_tags = []
    for sent_i in range(len(data)):
        sent = data[sent_i]

        # Fit the sentence length to the timestep dim if necessary
        if len(sent) > timestep:
            sent = sent[:timestep]
        elif len(sent) < timestep:
            sent.extend(
                [(constants.PAD_TOKEN, constants.WJS_NONE_TAG)] * (
                    timestep - len(sent)))

        sentence = []
        tags = []
        for token, tag in sent:
            char_embeddings = np.zeros((char_embed_dim, max_word_size),
                                       dtype=np.float32)

            if token != constants.PAD_TOKEN:
                # Convert char to embedded mappings
                rand_val = np.sqrt(3 / char_embed_dim)
                chr_vecs = np.random.uniform(-rand_val, rand_val,
                                             (char_embed_dim, len(token)))

                # Trim if the max word size is exceeded
                if chr_vecs.shape[1] > max_word_size:
                    char_embeddings = chr_vecs[:, :max_word_size]

                # Put the vector in the center if it doesn't fit the frame
                elif chr_vecs.shape[1] < max_word_size:
                    pad_size = (max_word_size - chr_vecs.shape[1]) // 2
                    for i in range(chr_vecs.shape[1]):
                        char_embeddings[:, i + pad_size] = chr_vecs[:, i]

                else:
                    char_embeddings = chr_vecs

            assert char_embeddings.shape[
                       0] == char_embed_dim, "Character embeddings dimensions do not match"
            assert char_embeddings.shape[
                       1] == max_word_size, "Maximum word size dimensions do not match"

            sentence.append(np.array(char_embeddings))
            tags.append(_item_to_onehot(tag, tag_ids))

        np_sent = np.array(sentence)
        assert np_sent.shape[0] == timestep, "Timestep dimensions do not match"
        new_sents.append(np_sent)

        np_tags = np.array(tags)
        assert np_tags.shape[1] == len(tag_ids)
        new_tags.append(np_tags)

    # Export to pickle file
    utils.export_pickle(constants.WJS_DATA,
                        "wjs_treebank_char_embedding_" + str(
                            max_word_size),
                        np.array(new_sents))
    utils.export_pickle(constants.WJS_DATA,
                        "wjs_treebank_pos_tags_one_hot_" + str(
                            max_word_size),
                        np.array(new_tags))


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

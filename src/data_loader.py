"""
Parsing datasets
"""
import os

import nltk

import constants
import input_process
import utils


def parse_WJS(data_dir):
    """
    Parses the WJS treebank dataset
    :param data_dir:
    :return: List of sentences where each sentence is represented as a
    (token, pos_tag) list
    """
    parsed_sents = []
    avg = 0
    max_len = 0
    reader = nltk.corpus.BracketParseCorpusReader(data_dir, ".*mrg")
    sentences = reader.parsed_sents()
    for sent in sentences:
        new_sent = [(s.leaves()[0], s.label()) for s in
                    sent.subtrees(lambda t: t.height() == 2)]
        parsed_sents.append(new_sent)
        avg += len(new_sent)
        max_len = max([max_len, len(new_sent)])
    print("Average sentence length: ", avg // len(sentences))
    print("Maximum sentence length: ", max_len)
    return parsed_sents


def parse_NER(data_dir):
    """
    Parses the NER ConLL 2003 dataset
    :param data_dir: Directory containing train, development and test data-sets
    :return: List of sentences where each sentence is represented as a
    (token, entity_tag) list
    """
    train = os.path.join(data_dir, "eng.train.txt")
    devel = os.path.join(data_dir, "eng.testa.txt")
    test = os.path.join(data_dir, "eng.testb.txt")

    def _parse_file(lines):
        parsed = []
        current_sentence = []
        for i in range(len(lines)):
            if "-DOCSTART-" in lines[i]:
                pass

            elif lines[i] != "\n":
                split = lines[i].rstrip().split(" ")
                token = split[0]
                tag = split[-1].split("-")[-1]
                current_sentence.append((token, tag))
            else:
                if current_sentence:
                    parsed.append(current_sentence)
                current_sentence = []

        return parsed

    with open(train, "r") as f:
        lines = f.readlines()
        parsed_train = _parse_file(lines)

    with open(devel, "r") as f:
        lines = f.readlines()
        parsed_devel = _parse_file(lines)

    with open(test, "r") as f:
        lines = f.readlines()
        parsed_test = _parse_file(lines)

    # Join training and development sets
    parsed_train.extend(parsed_devel)
    return parsed_train, parsed_test


def prepare_wjs_data(process_data, test_size):
    """
    Prepares the WJS Treebank dataset for training. Pipeline consists of
    embeddings creation and one-hot label encoding.
    :param process_data:
    :param test_size:
    :return:
    """
    # Setup paths
    path_prefix = os.path.join(constants.WJS_DATA,
                               "wjs_treebank_timest" + str(
                                   constants.TIMESTEP) + "_")
    word_embeddings_path = path_prefix + "word_embeddings.pkl"
    char_id_mappings_path = path_prefix + "char_id_mappings.pkl"
    char_embeddings_path = path_prefix + "char_embeddings.pkl"
    labels_path = path_prefix + "onehot_labels.pkl"

    if process_data:
        wjs_data = parse_WJS(constants.WJS_DATA_DIR)
        input_process.embed_words(wjs_data, export_file=word_embeddings_path)
        input_process.create_char_mappings(wjs_data,
                                           export_file=char_embeddings_path,
                                           mappings_export_file=char_id_mappings_path)
        input_process.encode_labels(wjs_data, export_file=labels_path)

    # Load data from disk
    treebank = utils.load_pickle(word_embeddings_path)
    pos_tags = utils.load_pickle(labels_path)
    chr_embds = utils.load_pickle(char_embeddings_path)
    chr_id_mappings = utils.load_pickle(char_id_mappings_path)

    # Shuffle
    chr_embds, treebank, pos_tags = utils.shuffle_data(chr_embds, treebank,
                                                       pos_tags)

    # Split
    train_chr, valid_chr, train_word, valid_word, train_label, \
    valid_label = utils.split_data(chr_embds, treebank, pos_tags,
                                   test_size=test_size)

    return chr_id_mappings, train_chr, valid_chr, train_word, valid_word, \
           train_label, valid_label, chr_id_mappings


def prepare_ner_data(process_data, test_size):
    """
    Prepares the NER dataset for training. Pipeline consists of
    embeddings creation and one-hot label encoding.
    :param process_data: Whether to process raw data
    :param test_size: Validation set size in percentages
    :return:
    """
    path_prefix = os.path.join(constants.NER_DATA,
                               "ner_conell_" + str(
                                   constants.TIMESTEP) + "_")
    word_embeddings_path = path_prefix + "word_embeddings.pkl"
    char_id_mappings_path = path_prefix + "char_id_mappings.pkl"
    char_embeddings_path = path_prefix + "char_embeddings.pkl"
    labels_path = path_prefix + "onehot_labels.pkl"

    if process_data:
        train_raw, test_raw = parse_NER(
            os.path.join(constants.NER_DATA, "raw", "train"))
        input_process.embed_words(train_raw, export_file=word_embeddings_path)
        input_process.create_char_mappings(train_raw,
                                           export_file=char_embeddings_path,
                                           mappings_export_file=char_id_mappings_path)
        input_process.encode_labels(train_raw, export_file=labels_path)

    # Load data from disk
    ner = utils.load_pickle(word_embeddings_path)
    pos_tags = utils.load_pickle(labels_path)
    chr_embds = utils.load_pickle(char_embeddings_path)
    chr_id_mappings = utils.load_pickle(char_id_mappings_path)

    train_chr, valid_chr, train_word, valid_word, train_label, \
    valid_label = utils.split_data(chr_embds, ner, pos_tags,
                                   test_size=test_size)

    return chr_id_mappings, train_chr, valid_chr, train_word, valid_word, \
           train_label, valid_label, chr_id_mappings

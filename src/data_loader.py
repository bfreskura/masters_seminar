"""
Parsing datasets
"""

import nltk


def parse_WJS(data_dir):
    """
    Parses the WJS treebank dataset
    :param data_dir:
    :return:
    """
    parsed_sents = []
    reader = nltk.corpus.BracketParseCorpusReader(data_dir, ".*mrg")
    sentences = reader.parsed_sents()
    for sent in sentences:
        new_sent = [(s.leaves()[0], s.label()) for s in
                    sent.subtrees(lambda t: t.height() == 2)]
        parsed_sents.append(new_sent)
    return parsed_sents

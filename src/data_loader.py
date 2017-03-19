"""
Parsing datasets
"""

import nltk
from nltk.corpus import treebank
import IPython


def parse_WJS(data_dir):
    """
    Parsed the WJS treebank
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

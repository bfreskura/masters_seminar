import os
import numpy as np


ROOT = os.path.dirname(__file__)

DATA = os.path.join(ROOT, "data")
NER_DATA = os.path.join(DATA, "ner")
WJS_DATA = os.path.join(DATA, "wjs_treebank")

RESOURCES = os.path.join(ROOT, "resources")

# Original files
WJS_DATA_DIR = "/home/bartol/nltk_data/corpora/treebank/combined"

# LSTM params

# Average sentence size
TIMESTEP = 25
MAX_WORD_SIZE = 20

# Dataset and resources
GLOVE_WIKI = "http://nlp.stanford.edu/data/glove.6B.zip"

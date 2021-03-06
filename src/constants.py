import os
import numpy as np

ROOT = os.path.dirname(__file__)

DATA = os.path.join(ROOT, "data")
NER_DATA = os.path.join(DATA, "ner")
WJS_DATA = os.path.join(DATA, "wjs_treebank")
CONFIGS = os.path.join(ROOT, "configs")

RESOURCES = os.path.join(ROOT, "resources")
TF_WEIGHTS = os.path.join(RESOURCES, "weights")
LOGS = os.path.join(RESOURCES, "logs")

# Original files
WJS_DATA_DIR = "/home/bartol/nltk_data/corpora/treebank/combined"

# LSTM params

TIMESTEP = 25
MAX_WORD_SIZE = 20
CHAR_EMBEDDINGS_FEATURE = 30

# Dataset and resources
GLOVE_WIKI = "http://nlp.stanford.edu/data/glove.6B.zip"


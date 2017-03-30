import os
import zipfile
import pickle
import constants
import wget
import numpy as np
import configparser


def export_pickle(dir, object):
    """
    Exports the python object to a pickle file
    :param dir: Export dir
    :param name: Exported file name without the extension
    :param object: Python object
    :return:
    """
    with open(os.path.join(dir), mode="wb") as f:
        pickle.dump(file=f, obj=object)
        print("Pickle exported to", os.path.join(dir))


def shuffle_data(chr_embds, word_embds, labels):
    """
    Shuffles Char embeddings, word embeddings and labels
    :param word_embds: Word embeddigns
    :param labels: One hot labels
    :return: Shuffled triple
    """
    indices = np.arange(len(word_embds))
    np.random.shuffle(indices)
    return chr_embds[indices], word_embds[indices], labels[indices]


def load_pickle(path):
    """
    Load the pickle file
    :param path:
    :return:
    """
    with open(path, mode="rb") as f:
        return pickle.load(f)


def dir_creator(dirs_list):
    """
    Creates directories if they don't exist.
    :param dirs_list: List of absolute directory paths
    :return:
    """
    for d in dirs_list:
        if not os.path.exists(d):
            os.makedirs(d)
            print("Created directory", d)


def extract_zip(file, ext_dir):
    """
    Extracts the zip file to a chosen directroy
    :param file: Zip file path
    :param ext_dir: Extraction directory
    :return:
    """
    print("Extracting", file, "to", ext_dir)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(ext_dir)
    print("Extraction finished!\n")


def download_data(download_dir="/tmp"):
    """
    Downloads the dataset and resources required for the project.
    NOTE: You should have at least 5GB of disk space available.
    :return:
    """
    glove_dir = os.path.join(constants.RESOURCES, "glove")
    tf_weights = os.path.join(constants.RESOURCES, "tf_weights")

    dir_creator(
        [constants.DATA, constants.RESOURCES, glove_dir, tf_weights])

    # Twitter glove vectors
    glove_name = os.path.join(download_dir, "glove.zip")
    if not os.path.exists(glove_name):
        print("Downloading Wikipedia Glove vector from", constants.GLOVE_WIKI)
        print("This may take a while because the file size is 1.4GB")
        wget.download(constants.GLOVE_WIKI, glove_name)
        print("Downloaded to", glove_name, "\n")
    extract_zip(glove_name, glove_dir)


def read_config(config_file):
    """
    Reads the configuration file and creates a dictionary
    :param config_file: Ini file path
    :return:
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    conf_dict = dict()

    conf_dict['lr'] = float(config["MODEL"]['learning rate'])
    conf_dict['optimizer'] = config["MODEL"]['optimizer']
    conf_dict['timestep'] = int(config["MODEL"]['timestep'])
    conf_dict['word_vector_dim'] = int(
        config["MODEL"]['word embedding dimension'])
    conf_dict['char_embeddings_dim'] = int(
        config["MODEL"]['character embeddings dimension'])
    conf_dict['max_word_size'] = int(config["MODEL"]['max word length'])
    conf_dict['filter_dim'] = int(config["MODEL"]['cnn filter dimension'])
    conf_dict['lstm_hidden'] = int(config["MODEL"]['lstm hidden state dim'])
    conf_dict['batch_size'] = int(config["MODEL"]['batch size'])
    conf_dict['domain'] = config["GENERAL"]['domain']
    conf_dict['train_epochs'] = int(config["GENERAL"]['training epochs'])

    return conf_dict


def split_data(chr_embds, treebank, pos_tags, test_size=0.3):
    """
    Split the dataset to training and development datasets
    :param chr_embds:
    :param treebank:
    :param pos_tags:
    :param test_size:
    :return:
    """
    ts = int((1 - test_size) * chr_embds.shape[0])

    train_chr, valid_chr = chr_embds[:ts], chr_embds[ts:]
    train_word, valid_word = treebank[:ts], treebank[ts:]
    train_label, valid_label = pos_tags[:ts], pos_tags[ts:]

    return train_chr, valid_chr, train_word, valid_word, train_label, valid_label

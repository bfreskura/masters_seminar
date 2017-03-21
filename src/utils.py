import os
import zipfile
import pickle
import constants
import wget


def export_pickle(dir, name, object):
    """
    Exports the python object to a pickle file
    :param dir: Export dir
    :param name: Exported file name without the extension
    :param object: Python object
    :return:
    """
    with open(os.path.join(dir, name + ".pkl"), mode="wb") as f:
        pickle.dump(file=f, obj=object)


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

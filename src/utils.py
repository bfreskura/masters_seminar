import os
import pickle


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

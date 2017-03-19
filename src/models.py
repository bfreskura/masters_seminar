import tensorflow as tf


class BLSTM_CNN():
    def __init__(self, config):
        self.learning_rate = config["lr"]
        self.optimizer = config["optimizer"]

"""
Load the test set and a saved model, and evaluate the test set with the model.
Print the resulting confusion matrix.
"""

import os
import pickle
import sys
from pathlib import Path

from logger import get_logger, config_thr_exc_log

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


def load_model_from(path: Path):
    """
    Load a saved model from the given directory.
    """

    assert path.is_dir(), f"Cannot load from non-existent directory {path}"
    return tf.keras.models.load_model(str(path))


if __name__ == "__main__":
    path = Path(sys.argv[1])
    model = load_model_from(path)

    x_test = np.load("./tensor/test/x_tensor.npy")
    author_test = np.load("./tensor/test/user_tensor.npy")
    topic_test = np.load("./tensor/test/topic_tensor.npy")

    with open("./input_data/test/y.p", "rb") as y_test_file:
        y_test = pickle.load(y_test_file)

    prediction = model.predict([x_test, author_test, topic_test], y_test)

    cnf_matrix = confusion_matrix(y_test, prediction)
    print(cnf_matrix)

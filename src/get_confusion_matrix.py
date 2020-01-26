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


def load_model_from(path: Path):
    """
    Load a saved model from the given directory.
    """

    assert path.is_dir(), f"Cannot load from non-existent directory {path}"
    return tf.keras.models.load_model(str(path))


def get_confusion_matrix(model, x_test, author_test, topic_test, y_test):
    prediction = model.predict([x_test, author_test, topic_test])
    return tf.math.confusion_matrix(y_test, np.around(prediction).astype(int))


if __name__ == "__main__":
    print("-- Loading data")
    x_test = np.load("./tensor/test/x_tensor.npy")
    author_test = np.load("./tensor/test/user_tensor.npy")
    topic_test = np.load("./tensor/test/topic_tensor.npy")

    with open("./input_data/test/y.p", "rb") as y_test_file:
        y_test = pickle.load(y_test_file)

    print("-- Data loaded")

    for model_path in Path('.').glob('model-tcntest-a*-1/modelsave'):
        model_name = model_path.parent.name
        print(f"-- Loading model {model_name}")
        model = load_model_from(model_path)

        cnf_matrix = get_confusion_matrix(model, x_test, author_test, topic_test, y_test)
        m = cnf_matrix.numpy()

        print(f"== Confusion matrix for {model_name} ==")
        print(f"{m[0,0]}\t{m[0,1]}\n{m[1,0]}\t{m[1,1]}\n")

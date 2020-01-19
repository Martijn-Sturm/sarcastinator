import pickle
import numpy as np
import os

from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from contextlib import redirect_stdout

from logger import get_logger
from logger import config_thr_exc_log

# from temporal import tcn1  # , tcn2
from orig_cnn import cnn2  # ,cnn1

# Loggers
logger = get_logger(filename="run_tune", name=__name__)
config_thr_exc_log()

# Params:
batch_size = 4000
epochs = 50

# ====================== LOAD DATA ====================================

# # Real data:
# logger.warning("Loading train data...")
# x_train = pickle.load(open("./tensor/train/x_tensor.p", "rb"))
# author_train = pickle.load(open("./tensor/train/user_tensor.p", "rb"))
# topic_train = pickle.load(open("./tensor/train/topic_tensor.p", "rb"))

# logger.info(f"x_train shape: {x_train.shape}")
# logger.info(f"author_train shape: {author_train.shape}")
# logger.info(f"topic_train shape: {topic_train.shape}")

# # Labels
# y_train = pickle.load(open("./tensor/train/y.p", "rb"))
# logger.info(f"y_train shape: {y_train.shape}")
# logger.warning("Loading data is finished")

# # shuffle data
# sample_set = set([
#     x_train.shape[0],
#     y_train.shape[0],
#     author_train.shape[0],
#     topic_train.shape[0]])

# if len(sample_set) > 0:
#     raise Exception("Different sample sizes for input features")

# shuffle_indices = np.random.permutation(np.arange(len(y_train)))
# x_train = x_train[shuffle_indices]
# y_train = y_train[shuffle_indices]
# author_train = author_train[shuffle_indices]
# topic_train = topic_train[shuffle_indices]

# Fake data:
# Sentences, words per sentence, dimensions per word for embedding
sample_size = 50
x_train = np.random.normal(size=(sample_size, 100, 300))
topic_train = np.random.normal(size=(sample_size, 100))
author_train = np.random.normal(size=(sample_size, 100))
y_train = np.random.randint(low=0, high=2, size=sample_size)


# =========================== BUILD MODEL ===================================


# Input shape:
input_shape_x = (x_train.shape[1], x_train.shape[2])  # words/sentence, word_emb dimensions
input_shape_author = (author_train.shape[1],)  # emb dimensions
input_shape_topic = (topic_train.shape[1],)  # emb dimensions
logger.info(f'Input shape x: {input_shape_x}')
logger.info(f'Input shape author: {input_shape_author}')
logger.info(f'Input shape topic: {input_shape_topic}')


# Load model
logger.warning("Creating model...")

# # tcn model
# model = tcn1(input_shape=input_shape_x, logger=logger)

# cnn model with merging of features
model = cnn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger)

logger.warning("Model created.")

# Determine number of params
mod_params = model.count_params()
logger.warning(f"Model params: {mod_params}")

# Save model summary
with open('./logs/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Callbacks:
callbacks = [
    # tf.keras.callbacks.EarlyStopping(
    #     # Stop training when `val_loss` is no longer improving
    #     monitor='val_loss',
    #     # "no longer improving" being defined as "no better than 1e-2 less"
    #     min_delta=1e-2,
    #     # "no longer improving" being further defined as "for at least 2 epochs"
    #     patience=2,
    #     verbose=1),
    tf.keras.callbacks.ProgbarLogger(
        count_mode='samples', stateful_metrics=None),
    tf.keras.callbacks.BaseLogger(
        stateful_metrics=None)
]

# Run model fitting
logger.warning("Model training is initiated...")
result = model.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")

accs_train = result.history['accuracy']
accs_val = result.history['val_accuracy']
logger.info(f"Training accuracy per epoch: {accs_train}")
logger.info(f"Validation accuracy per epoch: {accs_val}")

loss_train = result.history['loss']
loss_val = result.history['val_loss']
logger.info(f"Training accuracy per epoch: {loss_train}")
logger.info(f"Validation accuracy per epoch: {loss_val}")
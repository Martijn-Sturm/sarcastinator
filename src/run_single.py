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

from temporal import tcn1

# Loggers
logger = get_logger(filename="single_run", name=__name__)
config_thr_exc_log()

# Params:
batch_size = 4000
epochs = 10

# Real data:
# Rows=words, columns=embedding dimensions
word_embs = pickle.load(open("./input_data/train/word_embs.p", "rb"))
# Rows=sentences, columns=word indices
x_train = pickle.load(open("./input_data/train/x.p", "rb"))

# DEBUG CODE:
word_embs_type = str(type(word_embs))
logger.debug(f"word_embs object type: {word_embs_type}")
x_train_type = str(type(x_train))
logger.debug(f"x_train object type: {x_train_type}")
word_embs_dtype = str(word_embs.dtype)
logger.debug(f"dtype of word_embs: {word_embs_dtype}")
x_train_dtype = str(x_train.dtype)
logger.debug(f"dtype of x_train: {x_train_dtype}")
x_train_dtype = str(x_train.dtype)
logger.debug(f"dtype of x_train: {x_train_dtype}")
x_train = tf.cast(x_train, dtype=tf.int64)
logger.debug(f"dtype of x_train after cast: {x_train_dtype}")


logger.info(f"word_embs shape: {word_embs.shape}")
logger.info(f"x_train shape: {x_train.shape}")

# Create 3d tensor of shape [sentences,words,embeddingdims]
x_train = tf.nn.embedding_lookup(
    params=word_embs, ids=x_train, max_norm=None, name=None)
logger.info(f"x_train shape after embedding lookup: {x_train.shape}")

y_train = pickle.load(open("./input_data/train/y.p", "rb"))
logger.info(f"y_train shape: {y_train.shape}")

# # Fake data:
# # Sentences, words per sentence, dimensions per word for embedding
# x_train = np.random.normal(size=(200, 100, 300))
# y_train = np.random.randint(low=0, high=2, size=200)

# Input shape:
input_shape = (x_train.shape[1], x_train.shape[2])  # words/sentence, word_emb dimensions
logger.info(f'Input shape x: {input_shape}')

# Load model
logger.warning("Creating model...")
model = tcn1(input_shape=input_shape, logger=logger)
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
    x_train, y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks)
logger.warning("Model training is finished.")

accs = result.history['accuracy']
logger.info(f"Accuracy per epoch: {accs}")

# Evaluation
logger.warning("Evaluation is initiated")

# Rows=words, columns=embedding dimensions
word_embs = pickle.load(open("./input_data/test/word_embs.p", "rb"))
# Rows=sentences, columns=word indices
x_test = tf.cast(pickle.load(open("./input_data/test/x.p", "rb")), dtype=tf.int64)
logger.info(f"x_test shape: {x_test.shape}")
# Create 3d tensor of shape [sentences,words,embeddingdims]
x_test = tf.nn.embedding_lookup(
    params=word_embs, ids=x_test, max_norm=None, name=None)
logger.info(f"x_test shape after embedding lookup: {x_test.shape}")
y_test = pickle.load(open("./input_data/test/y.p", "rb"))
logger.info(f"y_test shape: {y_test.shape}")

eval_result = model.evaluate(
    x_test, y_test, batch_size=batch_size
)
logger.warning("Evaluation ended.")

logger.warning(f"evaluation loss: {eval_result[0]}")
logger.warning(f"evaluation accuracy: {eval_result[1]}")

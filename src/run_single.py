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
batch_size = 50
epochs = 10

# # Rows=words, columns=embedding dimensions
# word_embs = pickle.load(open(""./input_data/train/word_embs.p"", "rb"))
# # Rows=sentences, columns=word indices
# x_train = pickle.load(open("./input_data/train/x.p", "rb"))
# # Create 3d tensor of shape [sentences,words,embeddingdims]
# x_train = tf.nn.embedding_lookup(params=word_embs, ids=x, max_norm=None, name=None)
# y_train = pickle.load(open("./input_data/train/y.p", "rb"))

# Fake data:
# Sentences, words per sentence, dimensions per word for embedding
x_train = np.random.normal(size=(200, 100, 300))
y_train = np.random.randint(low=0, high=2, size=200)

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

# # Rows=words, columns=embedding dimensions
# word_embs = pickle.load(open(""./input_data/test/word_embs.p"", "rb"))
# # Rows=sentences, columns=word indices
# x_test = pickle.load(open("./input_data/test/x.p", "rb"))
# # Create 3d tensor of shape [sentences,words,embeddingdims]
# x_test = tf.nn.embedding_lookup(params=word_embs, ids=x, max_norm=None, name=None)
# y_test = pickle.load(open("./input_data/test/y.p", "rb"))
eval_result = model.evaluate(
    x_train, y_train, batch_size=batch_size
)
logger.warning("Evaluation ended.")

logger.warning(f"evaluation loss: {eval_result[0]}")
logger.warning(f"evaluation accuracy: {eval_result[1]}")

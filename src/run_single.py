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

from temporal import tcn1

logger = get_logger(name=__name__)

# # Rows=words, columns=embedding dimensions
# word_embs = pickle.load(open(""./input_data/word_embs.p"", "rb"))
# # Rows=sentences, columns=word indices
# x = pickle.load(open("./input_data/x.p", "rb"))
# # Create 3d tensor of shape [sentences,words,embeddingdims]
# x_train = tf.nn.embedding_lookup(params=word_embs, ids=x, max_norm=None, name=None)
# y_train = pickle.load(open("./input_data/y.p", "rb"))

# Fake data:
# Sentences, words per sentence, dimensions per word for embedding
x_train = np.random.normal(size=(200, 100, 300))
y_train = np.random.randint(low=0, high=2, size=200)

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

# Run model fitting
logger.warning("Model training is initiated...")
result = model.fit(x_train, y_train, batch_size=1000, epochs=10, verbose=1)
logger.warning("Model training is finished.")

print(result)

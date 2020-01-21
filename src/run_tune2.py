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

import pandas as pd

from logger import get_logger
from logger import config_thr_exc_log

from results_train import proc_results

from temporal import tcn2  # , tcn2
# from orig_cnn import cnn2  # ,cnn1

# Loggers
logger = get_logger(filename="run_tune2", name=__name__)
config_thr_exc_log()

# Params:
batch_size = 4000
epochs = 100
seed = 5

# ====================== LOAD DATA ====================================

# Real data:
logger.warning("Loading train data...")
try:
    x_train = np.load('./tensor/train/x_tensor.npy')
    author_train = np.load('./tensor/train/user_tensor.npy')
    topic_train = np.load('./tensor/train/topic_tensor.npy')
except FileNotFoundError:
    # X
    x_embs = pickle.load(open("./input_data/embs/word_embs.p", "rb"))
    x_idx = pickle.loadopen("./input_data/train/x.p", "rb")()
    x_train = tf.nn.embedding_lookup(
        params=x_embs, ids=x_idx)
    # Author
    author_embs = pickle.load(open("./input_data/embs/user_embs.p", "rb"))
    author_idx = pickle.loadopen("./input_data/train/author_train.p", "rb")()
    author_train = tf.nn.embedding_lookup(
        params=author_embs, ids=author_idx)
    # Topic
    topic_embs = pickle.load(open("./input_data/embs/topic_embs.p", "rb"))
    topic_idx = pickle.loadopen("./input_data/train/topic_train.p", "rb")()
    topic_train = tf.nn.embedding_lookup(
        params=topic_embs, ids=topic_idx)

logger.info(f"x_train shape: {x_train.shape}")
logger.info(f"author_train shape: {author_train.shape}")
logger.info(f"topic_train shape: {topic_train.shape}")

# Labels
y_train = pickle.load(open("./input_data/train/y.p", "rb"))
logger.info(f"y_train shape: {y_train.shape}")
logger.warning("Loading data is finished")

# shuffle data
sample_set = set([
    x_train.shape[0],
    y_train.shape[0],
    author_train.shape[0],
    topic_train.shape[0]])

if len(sample_set) > 1:
    raise Exception("Different sample sizes for input features")

np.random.seed(seed)
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]
author_train = author_train[shuffle_indices]
topic_train = topic_train[shuffle_indices]

# # Fake data:
# # Sentences, words per sentence, dimensions per word for embedding
# sample_size = 10
# x_train = np.random.normal(size=(sample_size, 100, 300))
# topic_train = np.random.normal(size=(sample_size, 100))
# author_train = np.random.normal(size=(sample_size, 100))
# y_train = np.random.randint(low=0, high=2, size=sample_size)


# =========================== BUILD MODEL ===================================


# Input shape:
input_shape_x = (x_train.shape[1], x_train.shape[2])  # words/sentence, word_emb dimensions
input_shape_author = (author_train.shape[1],)  # emb dimensions
input_shape_topic = (topic_train.shape[1],)  # emb dimensions
logger.info(f'Input shape x: {input_shape_x}')
logger.info(f'Input shape author: {input_shape_author}')
logger.info(f'Input shape topic: {input_shape_topic}')

# Callbacks:
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=3,
        verbose=1),
    tf.keras.callbacks.ProgbarLogger(
        count_mode='samples', stateful_metrics=None),
    tf.keras.callbacks.BaseLogger(
        stateful_metrics=None)
]
ls_res_dicts = []

# cnn model with merging of features
logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, filter_sizes=[2],
    l2_float=0.001, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train],
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-filt2", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, filter_sizes=[3], learn_rate=0.0005,
    l2_float=0.001, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train],
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-filt3LR", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, filter_sizes=[4],
    l2_float=0.001, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train],
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-filt4", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.001, dropout=0.2, filter_sizes=(2, 3, 4))
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-filtall", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.001, dropout=0.2,
    num_filters=64)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-numfilt64", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.001, dropout=0.2,
    num_filters=128)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-numfilt128", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, filter_sizes=[2],
    l2_float=0.001, dropout=0.2,
    num_filters=64)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs, 
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-2numfilt64", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, filter_sizes=[2],
    l2_float=0.001, dropout=0.2,
    num_filters=128)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs, 
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-2numfilt128", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, filter_sizes=[3],
    l2_float=0.001, dropout=0.2,
    num_filters=64)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs, 
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-3numfilt64", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, filter_sizes=[3],
    l2_float=0.001, dropout=0.2,
    num_filters=128)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs, 
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-3numfilt128", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, filter_sizes=[4],
    l2_float=0.001, dropout=0.2,
    num_filters=64)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs, 
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-4numfilt64", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, filter_sizes=[4],
    l2_float=0.001, dropout=0.2,
    num_filters=128)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs, 
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-4numfilt128", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.001, dropout=0.2,
    num_filters=16)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-numfilt16", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.001, dropout=0.2,
    dense_layers=16)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dense16", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.001, dropout=0.2,
    dense_layers=64)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dense64", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.001, dropout=0.2,
    dense_layers=128)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dense128", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.001, dropout=0.5)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-drop05", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.01, dropout=0.5)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dropl2", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.01, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-l2", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.01, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=1000, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-batch1000", logger, param_dict1,
             **{"Batchsize":1000})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger,
    l2_float=0.01, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=10000, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-batch10000", logger, param_dict1,
             **{"Batchsize":10000})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

# from here on, skips=False
logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False, filter_sizes=[2],
    l2_float=0.001, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train],
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipfilt2", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False, filter_sizes=[3],
    learn_rate=0.0005, l2_float=0.001, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train],
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipfilt3LR", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False, filter_sizes=[3],
    l2_float=0.001, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train],
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipfilt4", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.001, dropout=0.2, filter_sizes=(2, 3, 4))
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipfiltall", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.001, dropout=0.2,
    num_filters=64)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipnumfilt64", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.001, dropout=0.2,
    num_filters=128)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipnumfilt128", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.001, dropout=0.2,
    num_filters=16)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipnumfilt16", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.001, dropout=0.2,
    dense_layers=16)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipdense16", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.001, dropout=0.2,
    dense_layers=64)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipdense64", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.001, dropout=0.2,
    dense_layers=128)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipdense128", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.001, dropout=0.5)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipdrop05", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.01, dropout=0.5)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipdropl2", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.01, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipl2", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.01, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=1000, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipbatch1000", logger, param_dict1,
             **{"Batchsize":1000})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, skips=False,
    l2_float=0.01, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=10000, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilskipbatch10000", logger, param_dict1,
             **{"Batchsize":10000})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

# From here, dilation correction
logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1, filter_sizes=[2],
    l2_float=0.001, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train],
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilfilt2", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1, filter_sizes=[3],
    learn_rate=0.0005, l2_float=0.001, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train],
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilfilt3LR", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1, filter_sizes=[3],
    l2_float=0.001, dropout=0.2)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train],
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilfilt4", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1,
    l2_float=0.001, dropout=0.2, filter_sizes=(2, 3, 4))
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilfiltall", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1,
    l2_float=0.001, dropout=0.2,
    num_filters=64)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilnumfilt64", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1,
    l2_float=0.001, dropout=0.2,
    num_filters=128)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilnumfilt128", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1,
    l2_float=0.001, dropout=0.2,
    num_filters=16)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dilnumfilt16", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1,
    l2_float=0.001, dropout=0.2,
    dense_layers=16)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dildense16", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1,
    l2_float=0.001, dropout=0.2,
    dense_layers=64)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dildense64", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1,
    l2_float=0.001, dropout=0.2,
    dense_layers=128)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dildense128", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1,
    l2_float=0.001, dropout=0.5)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dildrop05", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

logger.warning("Creating model...")
model1, param_dict1 = tcn2(
    input_shape_x=input_shape_x,
    input_shape_topic=input_shape_topic,
    input_shape_author=input_shape_author,
    logger=logger, dil_cor=1,
    l2_float=0.01, dropout=0.5)
logger.warning("Model created.")
logger.warning("Model training is initiated...")
result1 = model1.fit(
    [x_train, author_train, topic_train], 
    y_train, batch_size=batch_size, epochs=epochs,
    verbose=1, callbacks=callbacks, validation_split=0.1)
logger.warning("Model training is finished.")
res_dict = proc_results(result1, model1, "CNN-dildropl2", logger, param_dict1,
             **{"Batchsize":batch_size})
ls_res_dicts.append(res_dict)
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")

# Create df and write to csv file:
df = pd.DataFrame(ls_res_dicts)
df.to_csv("./results/dataframe.csv")
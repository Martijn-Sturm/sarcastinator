#!/usr/bin/env python
import pickle
import numpy as np
import os
import csv
import tensorflow as tf


from logger import get_logger
from logger import config_thr_exc_log

# Loggers
logger = get_logger(filename="prepare", name=__name__)
config_thr_exc_log()

logger.warning("================\nPreparing data initiated...\n================\n")
# ==================================== LOAD DATA ==================================================================

# ----------------- Word embedding comments -------------------------------
# Word embeddings for comments:
logger.warning("loading word embedding data...")
x = pickle.load(open("./mainbalancedpickle.p","rb"))
revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
logger.warning("word embedding data loaded!")
logger.info(f"Max length of sentences is: {max_l}")

# ------------------------- User features: ----------------------------------------------------
logger.warning("Starting user feature processing:")
logger.warning('loading wgcca embeddings...')
wgcca_embeddings = np.load('./../users/user_embeddings/user_gcca_embeddings.npz')
logger.warning('wgcca embeddings loaded')

# ids: Array len(283591)
# Add one value at begin of array 'unknown' ?
ids = np.concatenate((np.array(["unknown"]), wgcca_embeddings['ids']), axis=0)
# User embeddings: matrix
#  Shape is: (283591, 100)
# rows is users, columns is embedding dimensions
user_embeddings = wgcca_embeddings['G']
try:
    user_emb_shape = user_embeddings.shape
    logger.info(f"user embedding shape: {user_emb_shape}")
except Exception:
    logger.debug(f"Not possible to get shape of user embeddings and log")

unknown_vector = np.random.normal(size=(1, 100))
# Also add one embedding array to beginning of matrix randomly generated
user_embeddings = np.concatenate((unknown_vector, user_embeddings), axis=0)
user_embeddings = user_embeddings.astype(dtype='float32')
# So first rows of both ids and user_embeddings is for unknown author

# Make a dict with keys=author_name, value=index (corresponding to index of wgca embedding matrix)
wgcca_dict = {}
for i in range(len(ids)):
    wgcca_dict[ids[i]] = int(i)
logger.warning("User feature processing finished")

# -------------------------------- Discourse features --------------------------------------
logger.warning("Starting Topic feature processing")
csv_reader = csv.reader(open("./../discourse/discourse_features/discourse.csv"))
topic_embeddings = []
topic_ids = []
for line in csv_reader:
    topic_ids.append(line[0])
    topic_embeddings.append(line[1:])
topic_embeddings = np.asarray(topic_embeddings)
topic_embeddings = topic_embeddings.astype(dtype='float32')
try:
    topic_embeddings_size = len(topic_embeddings[0])
    logger.info(f"topic emb size: {topic_embeddings_size}")
except Exception:
    logger.debug("Not able to get and log topic embedding size")

topics_dict = {}
for i in range(len(topic_ids)):
    try:
        topics_dict[topic_ids[i]] = int(i)
    except TypeError:
        logger.error(f"Was not able to retrieve {i} in topic_ids")

# ???? Why change it to 100 ???
max_l = 100
logger.warning(f"Max_l was manually reset to: {max_l}")
logger.warning("Topic feature processing finished")
# ============================ Hash data ===============================

logger.warning("Starting with hashing of data")
# ------------- Initialize lists -----------------------------------------
# The following lists all have the same length. The index of the lists correspond to the same post_id (revs[i])

# List with all train data comment text
x_text = []
# list with wgca embedding indices
author_text_id = []
# Same but for topic embeddings
topic_text_id = []
# Label values
y = []

# Same for these lists:
test_x = []
test_topic = []
test_author = []
test_y = []

logger.warning("Filling data variables")
# -------------- Fill lists ------------------------------------------
# Loop over all post dicts in revs:
for i in range(len(revs)):
    # If post belongs to train data split
    if revs[i]['split']==1:
        x_text.append(revs[i]['text'])
        try:
            # append wgcca embedding index corresponding to current author to list
            author_text_id.append(wgcca_dict['"'+revs[i]['author']+'"'])
        except KeyError:
            author_text_id.append(0)
        try:
            # See above
            topic_text_id.append(topics_dict['"'+revs[i]['topic']+'"'])
        except KeyError:
            topic_text_id.append(0)
        # Label (sarcasm or not)
        temp_y = revs[i]['label']
        y.append(temp_y)
    # Else belongs to test data split
    else:
        test_x.append(revs[i]['text'])
        try:
            test_author.append(wgcca_dict['"'+revs[i]['author']+'"'])
        except:
            test_author.append(0)
        try:
            test_topic.append(topics_dict['"'+revs[i]['topic']+'"'])
        except:
            test_topic.append(0)
        test_y.append(revs[i]['label'])

y = np.asarray(y)
test_y = np.asarray(test_y)

# ------------------ get word embedding indices -------------------------
logger.warning("Process comments")
x = []
# For all posts:
for i in range(len(x_text)):
    # split the words in each post
    # and add the index of the word embedding for each word in that post:
    # To a list in list x
    x.append(np.asarray([word_idx_map[word] for word in x_text[i].split()]))
# So each list in x contains lists with the indices of the word embeddings of the words in that post/comment

x_test = []
for i in range(len(test_x)):
    x_test.append(np.asarray([word_idx_map[word] for word in test_x[i].split()]))

# ???? padding of short posts/comments  ???
# padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, padding='post')
# https://www.tensorflow.org/guide/keras/masking_and_padding
# Is this still necessary for Temporal Convolutional networks?
for i in range(len(x)):
    if(len(x[i]) < max_l):
        x[i] = np.append(x[i], np.zeros(max_l-len(x[i])))
    elif(len(x[i]) > max_l):
        x[i] = x[i][0:max_l]
x = np.asarray(x)
# After this step, every post contains the same number of word embeddings
# Which is max_l

for i in range(len(x_test)):
    if(len(x_test[i]) < max_l):
        x_test[i] = np.append(x_test[i], np.zeros(max_l-len(x_test[i])))
    elif(len(x_test[i]) > max_l):
        x_test[i] = x_test[i][0:max_l]
x_test = np.asarray(x_test)
y_test = test_y

topic_train = np.asarray(topic_text_id)
topic_test = np.asarray(test_topic)
author_train = np.asarray(author_text_id)
author_test = np.asarray(test_author)

# Process labels:
logger.warning("Transforming labels from 2d to 1d array")
# Train y:
if y.shape[1] != 2:
    raise Exception("Y has not expected dimensions",
                    "y shape:", y.shape)
y = y[:, 0]
if len(y.shape) != 1:
    raise Exception("New y has not 1 dimension. Shape:", y.shape)
# Test y:
if y_test.shape[1] != 2:
    raise Exception("Y has not expected dimensions",
                    "y shape:", y.shape)
y_test = y_test[:, 0]
if len(y_test.shape) != 1:
    raise Exception("New test_y has not 1 dimension. Shape:", y_test.shape)
logger.warning("Transforming labels finished")

# Check for decimals in lookup arrays:
# Train
x_dec = np.count_nonzero(np.modf(x)[0])
logger.info(f"non 0 decimals in x: {x_dec}")
y_dec = np.count_nonzero(np.modf(y)[0])
logger.info(f"non 0 decimals in y: {y_dec}")
author_train_dec = np.count_nonzero(np.modf(author_train)[0])
logger.info(f"non 0 decimals in author_train: {author_train_dec}")
topic_train_dec = np.count_nonzero(np.modf(topic_train)[0])
logger.info(f"non 0 decimals in topic_train: {topic_train_dec}")

ls_dec_train = [x_dec, y_dec, author_train_dec, topic_train_dec]
for i in ls_dec_train:
    if i > 0:
        raise Exception("Decimals found: ", i, "Check log for more info")

# Test
x_test_dec = np.count_nonzero(np.modf(x_test)[0])
logger.info(f"non 0 decimals in x: {x_test_dec}")
y_test_dec = np.count_nonzero(np.modf(y_test)[0])
logger.info(f"non 0 decimals in y: {y_test_dec}")
author_test_dec = np.count_nonzero(np.modf(author_test)[0])
logger.info(f"non 0 decimals in author_test: {author_test_dec}")
topic_test_dec = np.count_nonzero(np.modf(topic_test)[0])
logger.info(f"non 0 decimals in topic_test: {topic_test_dec}")

ls_dec_test = [x_test_dec, y_test_dec, author_test_dec, topic_test_dec]
for i in ls_dec_test:
    if i > 0:
        raise Exception("Decimals found: ", i, "Check log for more info")

# Casting:
# Train
x = x.astype(dtype="int64", casting='unsafe')
y = y.astype(dtype="int64", casting='unsafe')
author_train = author_train.astype(dtype="int64", casting='unsafe')
topic_train = topic_train.astype(dtype="int64", casting='unsafe')

# Test
x_test = x_test.astype(dtype="int64", casting='unsafe')
y_test = y_test.astype(dtype="int64", casting='unsafe')
author_test = author_test.astype(dtype="int64", casting='unsafe')
topic_test = topic_test.astype(dtype="int64", casting='unsafe')

# Write train data to pickles:
logger.warning("Writing train data to files")
os.makedirs("./input_data/train/")

pickle.dump(x, open("./input_data/train/x.p", "wb"))
pickle.dump(y, open("./input_data/train/y.p", "wb"))

pickle.dump(topic_train, open("./input_data/train/topic_train.p", "wb"))
pickle.dump(author_train, open("./input_data/train/author_train.p", "wb"))

# Write test data to pickles:
logger.warning("Writing test data to files")
os.makedirs("./input_data/test/")

pickle.dump(x_test, open("./input_data/test/x.p", "wb"))
pickle.dump(y_test, open("./input_data/test/y.p", "wb"))
pickle.dump(W, open("./input_data/test/word_embs.p", "wb"))
pickle.dump(topic_test, open("./input_data/test/topic_test.p", "wb"))
pickle.dump(author_test, open("./input_data/test/author_test.p", "wb"))

# Write embeddings to pickles:
logger.warning("Writing embeddings to files")
os.makedirs("./input_data/embs/")

pickle.dump(W, open("./input_data/embs/word_embs.p", "wb"))
pickle.dump(user_embeddings, open("./input_data/embs/user_embs.p", "wb"))
pickle.dump(topic_embeddings, open("./input_data/embs/topic_embs.p", "wb"))


# Create tensor imputs:
# Train
x_tensor_train = tf.nn.embedding_lookup(
    params=W, ids=x, max_norm=None, name=None)
logger.info(f"shape x_tensor_train {x_tensor_train.shape}")
author_tensor_train = tf.nn.embedding_lookup(
    params=user_embeddings, ids=author_train, max_norm=None, name=None)
logger.info(f"shape author_tensor_train {author_tensor_train.shape}")
topic_tensor_train = tf.nn.embedding_lookup(
    params=topic_embeddings, ids=topic_train, max_norm=None, name=None)
logger.info(f"shape topic_tensor_train {topic_tensor_train.shape}")

# Test
x_tensor_test = tf.nn.embedding_lookup(
    params=W, ids=x_test, max_norm=None, name=None)
logger.info(f"shape x_tensor_test {x_tensor_test.shape}")
author_tensor_test = tf.nn.embedding_lookup(
    params=user_embeddings, ids=author_test, max_norm=None, name=None)
logger.info(f"shape author_tensor_test {author_tensor_test.shape}")
topic_tensor_test = tf.nn.embedding_lookup(
    params=topic_embeddings, ids=topic_test, max_norm=None, name=None)
logger.info(f"shape topic_tensor_test {topic_tensor_test.shape}")

# Write embeddings to pickles:
logger.warning("Writing tensors to files")
os.makedirs("./tensor/train/")

np.save('./tensor/train/x_tensor.npy', x_tensor_train, allow_pickle=False)
np.save('./tensor/train/user_tensor.npy', author_tensor_train, allow_pickle=False)
np.save('./tensor/train/topic_tensor.npy', topic_tensor_train, allow_pickle=False)

os.makedirs("./tensor/test/")

np.save('./tensor/test/x_tensor.npy', x_tensor_test, allow_pickle=False)
np.save('./tensor/test/user_tensor.npy', author_tensor_test, allow_pickle=False)
np.save('./tensor/test/topic_tensor.npy', topic_tensor_test, allow_pickle=False)

logger.warning("\n===============\nPreparing finished!\n===============")
# # Fake data:
# x = np.array(range(1,61))
# y = np.array(range(61,121))
# topic_train = np.array(range(121, 181))
# author_train = np.array(range(181, 241))
# x_test = x
# y_test = y
# topic_test = topic_train
# author_test = author_train

# # Create folds indices:
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# NFOLD = 5
# leftover = (len(shuffle_indices) % NFOLD)
# if leftover != 0:
#     folds = np.split(
#         shuffle_indices[:-leftover],
#         NFOLD)
# else:
#     folds = np.split(
#         shuffle_indices,
#         NFOLD)

# fold1, fold2, fold3, fold4, fold5 = folds[0], folds[1], folds[2], folds[3], folds[4]


# def create_fold(fold, x, y, topic, author):
#     x_fold = x[fold]
#     y_fold = y[fold]
#     topic_fold = topic[fold]
#     author_fold = author[fold]

#     return([x_fold, y_fold, topic_fold, author_fold])

# fold_ls1 = create_fold(fold1, x, y, topic_train, author_train)
# fold_ls2 = create_fold(fold2, x, y, topic_train, author_train)
# fold_ls3 = create_fold(fold3, x, y, topic_train, author_train)
# fold_ls4 = create_fold(fold4, x, y, topic_train, author_train)
# fold_ls5 = create_fold(fold5, x, y, topic_train, author_train)

# fold_ls_test = [x_test, y_test, topic_test, author_test]

# os.makedirs("./folds/")

# pickle.dump(fold_ls1, open("./folds/fold_1.p", "wb"))
# pickle.dump(fold_ls1, open("./folds/fold_2.p", "wb"))
# pickle.dump(fold_ls1, open("./folds/fold_3.p", "wb"))
# pickle.dump(fold_ls1, open("./folds/fold_4.p", "wb"))
# pickle.dump(fold_ls1, open("./folds/fold_5.p", "wb"))

# pickle.dump(fold_ls1, open("./folds/fold_test.p", "wb"))

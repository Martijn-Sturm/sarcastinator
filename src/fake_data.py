import numpy as np 
import tensorflow as tf
from sklearn.datasets import make_blobs

def give_data(n, features):
    """
    """
    x, y = make_blobs(n_samples = n, centers = 2, n_features= features)
    return x, y

def give_data_x(sent, vocab, wordpersent, dims):
    from sklearn.datasets import make_blobs

    sent_word = np.random.randint(
        # Vocabulary: 50
        low = 1, high = vocab, 
        # 100 sentences with 20 words each
        size = (sent, wordpersent))

    embs = np.random.normal(size=(vocab, dims))

    
    # x, y = make_blobs(n_samples = n, centers = 2, n_features= features)
    return sent_word, embs

# test embedding lookup:
sent_word_matrix, word_embs = give_data_x(
    sent = 50,
    vocab = 400,
    wordpersent = 15,
    dims = 200
)

x_train_check = tf.nn.embedding_lookup(
    params=word_embs,
    ids=sent_word_matrix,
    max_norm=None,
    name=None
)
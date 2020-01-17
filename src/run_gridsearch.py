import pickle
import numpy as np
import os
# Keras / tensorflow imports
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
# Helper imports
from sklearn.model_selection import GridSearchCV
from contextlib import redirect_stdout
# Local imports
from logger import get_logger
# from temporal import tcn1
from tcn import TCN

# Activate logger
logger = get_logger()

# PARAMS:
CV = 3


# # ==================== LOAD DATA ========================
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

def tcn1(input_shape=input_shape, logger=logger, optimizer="Adam", init="he_normal"):

    # Calculate number of blocks:
    def calc_dilations(filter_size, field):
        import math
        max_dil = field / filter_size
        max_dil = math.ceil(math.log(max_dil) / math.log(2))
        dil_list = [2**i for i in range(0,max_dil+1)]
        return(dil_list)

    # TCN params
    nb_filters = 8
    filter_size = 2
    dilation_list = calc_dilations(
        filter_size=filter_size,
        field=input_shape[1])

    padding = "same"
    use_skip_connections = True
    dropout_rate = 0.0
    activation = "relu"
    kernel_initializer = "he_normal"
    use_batch_norm = True
    use_layer_norm = True
    nb_stacks = 1

    logger.info(f"Dilation list: {dilation_list}")

    # Input layer:
    i = Input(shape=input_shape)

    # Temporal convolutional layer
    o = TCN(
        nb_filters=nb_filters,
        kernel_size=filter_size,
        dilations=dilation_list,
        padding=padding,
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate,
        activation=activation,
        kernel_initializer=kernel_initializer,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        return_sequences=False,
        nb_stacks=nb_stacks
        )(i)

    # Output Layer
    o = Dense(1, activation=softmax)(o)

    model = Model(inputs=[i], outputs=[o])
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


# Create model
logger.warning("Creating model...")
model_CV = KerasClassifier(build_fn=tcn1, verbose=1)
model = tcn1()
logger.warning("Model created.")

# Determine number of parameters model
mod_params = model.count_params()
logger.warning(f"Model params: {mod_params}")

# Save model summary
with open('./logs/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# define the grid search parameters
# init_mode = ['he_normal']
batch_size = [1000, 5000]
epochs = [10]
# Load in param_grid
param_grid = dict(
    # init_mode=init_mode,
    batch_size=batch_size,
    epochs=epochs)

# Calculate number of params
n_params = 0
for lists in param_grid.values():
    n_params += len(lists)

# Log param and cross validation numbers
logger.warning(f"Model has number of hyperparamaters to tune: {n_params}")
logger.info(f"Model has the following CV parameter grid: {str(param_grid)}")
logger.warning(f"Number of folds for Cross validation is set to {CV}")

# Perform grid search
logger.warning("Grid search is initiated...")
grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=CV)
grid_result = grid.fit(x_train, y_train)
logger.warning("Grid search is finished.")


# ============= Reporting ==============================================
logger.warning(
    f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}\n')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    logger.info(f' mean={mean:.4}, std={stdev:.4} using {param}\n')

logger.warning("Run finished")

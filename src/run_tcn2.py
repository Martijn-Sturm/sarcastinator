import pickle
import numpy as np
import os 

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.utils import plot_model

import tensorflow as tf

from sklearn.model_selection import GridSearchCV

from tcn import TCN

# will be filename of text file
run_name = "run1"

# # Rows=words, columns=embedding dimensions
# word_embs = pickle.load(open(""./input_data/word_embs.p"", "rb"))
# # Rows=sentences, columns=word indices
# x = pickle.load(open("./input_data/x.p", "rb"))
# # Create 3d tensor of shape [sentences,words,embeddingdims]
# x_train = tf.nn.embedding_lookup(params=word_embs, ids=x, max_norm=None, name=None)
# y_train = pickle.load(open("./input_data/y.p", "rb"))

# Fake data:
# x_train, y_train = give_data(300, 20)
# Sentences, words per sentence, dimensions per word for embedding
x_train = np.random.normal(size = (200, 100, 300))
y_train = np.random.randint(low = 0, high = 2, size = 200)

input_shape = (x_train.shape[1], x_train.shape[2]) # words/sentence, word_emb dimensions

seed = 7
np.random.seed(seed)

# Calculate number of blocks:
def calc_dilations(filter_size, field):
    import math
    max_dil = field / filter_size
    max_dil = math.ceil(math.log(max_dil) / math.log(2))
    dil_list = [2**i for i in range(0,max_dil+1)]
    return(dil_list)


# ======== parameters for TCN =====================
# TCN params
nb_filters = 8
filter_size = 2
dilation_list = calc_dilations(
    filter_size = filter_size,
    field = x_train.shape[1]
)
padding = "same"
use_skip_connections = True
dropout_rate = 0.0
activation = "relu"
kernel_initializer = "he_normal"
use_batch_norm = True
use_layer_norm = True
nb_stacks = 1

print("Dilation list:")
print(dilation_list)

def create_model(optimizer="Adam", init="he_normal"):

    # Input layer:
    i = Input(shape = input_shape)

    # Temporal convolutional layer
    o = TCN(
    nb_filters= nb_filters,
    kernel_size = filter_size,
    dilations = dilation_list,
    padding = padding,
    use_skip_connections = use_skip_connections,
    dropout_rate = dropout_rate,
    activation = activation,
    kernel_initializer = kernel_initializer,
    use_batch_norm = use_batch_norm,
    use_layer_norm = use_layer_norm,
    return_sequences=False
    )(i)

    # Output Layer
    o = Dense(1, activation=softmax)(o)
    
    model = Model(inputs=[i], outputs=[o])
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    # history = model.fit(x_train, y_train, 
    #                     validation_data=[x_val, y_val],
    #                     batch_size=params['batch_size'],
    #                     epochs=params['epochs'],
    #                     verbose=0)
    
    # finally we have to make sure that history object and model are returned
    return model #, history


model = create_model()
print("Model params:", model.count_params())
print(model.summary())
# plot_model(model)
# result = model.fit(x_train, y_train, batch_size = 100, epochs = 10)

model_CV = KerasClassifier(
    build_fn=create_model)

# define the grid search parameters
# init_mode = ['he_normal']
batch_size = [1000, 5000]
epochs = [10]

param_grid = dict(
    # init_mode=init_mode,
    batch_size = batch_size,
    epochs = epochs)
grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=3, verbose = 0)
grid_result = grid.fit(x_train, y_train)


# ============= Reporting ==============================================
# Report in console:
print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f' mean={mean:.4}, std={stdev:.4} using {param}')

print("This info will also be written to a file")
# Report in txt file:
with open(f"report_{run_name}.txt", "w") as file:
    file.writelines(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}\n')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        file.writelines(f' mean={mean:.4}, std={stdev:.4} using {param}\n')

print(f"report_{run_name}.txt has been created in {os.getcwd()}")
print("Run finished")
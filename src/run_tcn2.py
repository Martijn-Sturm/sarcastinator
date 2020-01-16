# import pickle
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

def give_data(n, features):
    from sklearn.datasets import make_blobs

    x, y = make_blobs(n_samples = n, centers = 2, n_features= features)
    return x, y

# will be filename of text file
run_name = "run1"

# x_train = pickle.load(open("./input_data/x.p", "rb"))
# y_train = pickle.load(open("./input_data/y.p", "rb"))

x_train, y_train = give_data(300, 20)

input_shape = (1, 20) # Timesteps, dimensions

seed = 7
np.random.seed(seed)

# ======== parameters for TCN =====================
# TCN params
nb_filters = 32
filter_size = 2
dilation_list = (1,2,4,8,16,32)
padding = "same"
use_skip_connections = True
dropout_rate = 0.0
activation = "relu"
kernel_initializer = "he_normal"
use_batch_norm = True
use_layer_norm = True


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
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['Accuracy'])
    
    # history = model.fit(x_train, y_train, 
    #                     validation_data=[x_val, y_val],
    #                     batch_size=params['batch_size'],
    #                     epochs=params['epochs'],
    #                     verbose=0)
    
    # finally we have to make sure that history object and model are returned
    return model #, history

# plot_model(create_model)
model = create_model()

result = model.fit(x_train, y_train)

model_CV = KerasClassifier(build_fn=create_model, verbose=1)

# define the grid search parameters
init_mode = ['he_normal']
batch_size = [10]
epochs = [10]

param_grid = dict(
    init_mode=init_mode,
    batch_size = batch_size,
    epochs = epochs)
grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=3)
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
    file.writelines(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        file.writelines(f' mean={mean:.4}, std={stdev:.4} using {param}')

print(f"report_{run_name}.txt has been created in {os.getcwd()}")
print("Run finished")
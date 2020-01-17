from tcn import TCN

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def tcn1(input_shape, logger, optimizer="Adam", init="he_normal"):

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

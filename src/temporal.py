from tcn import TCN

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Concatenate, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l2

from logger import get_logger


def tcn1(input_shape, logger,
         nb_filters=128,
         filter_size=2,
         optimizer="Adam",
         init="he_normal"):

    # Calculate number of blocks:
    def calc_dilations(filter_size, field):
        import math
        max_dil = field / filter_size
        max_dil = math.ceil(math.log(max_dil) / math.log(2))
        dil_list = [2**i for i in range(0, max_dil+1)]
        return(dil_list)

    # TCN params
    nb_filters = nb_filters
    filter_size = filter_size
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
    o = Dense(1, activation="sigmoid")(o)

    model = Model(inputs=[i], outputs=[o])
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


def tcn2(input_shape_x, input_shape_author, input_shape_topic,
         logger, dense_layers=32, learn_rate=0.001,
         init="he_normal",
         num_filters=32, filter_sizes=[3], activation="relu",
         padding="same", l2_float=0.001, dropout=0.2,
         skips=True, num_stacks=2):
    """Convolution Network WITH merging of other feature vectors
    """
    optimizer = Adam(learning_rate=learn_rate)
    
    param_dict = {
        "num filters": num_filters,
        "filter sizes": filter_sizes,
        "padding": padding,
        "activation": activation,
        "n dense layer": dense_layers,
        "l2 regu": l2_float,
        "drop-out": dropout,
        "kernel init": init,
        "skip connections": skips,
        "n stacks": num_stacks,
        "learn rate": learn_rate
    }
    # Check if the right shape was passed as argument
    if len(input_shape_x) != 2:
        raise Exception("Input shape needs to have 2 dimensions.",
                        "input shape:", input_shape_x)

    logger.info(f"input shape post (word/sent, word_emb_dims): {input_shape_x}")

    # Input layer
    i_x = Input(shape=input_shape_x)  # (words/sent, word_emb)
    # embedding_size = input_shape[1]

    # Calculate number of blocks:
    def calc_dilations(filter_size, field, stacks):
        import math
        max_dil = field / filter_size / stacks
        max_dil = math.ceil(math.log(max_dil)/math.log(2))
        dil_list = [2**i for i in range(0, max_dil+1)]
        return(dil_list)

    # Convolutional layes
    logger.info(f"Filter sizes: {filter_sizes}")
    tcn_compl = []
    for filter_size in filter_sizes:

        # Determine dilation list:
        dilation_list = calc_dilations(
            filter_size=filter_size,
            field=input_shape_x[0],
            stacks=num_stacks)
        logger.info(f"Dilation list: {dilation_list}")

        o_tcn = TCN(
            nb_filters=num_filters,
            kernel_size=filter_size,
            dilations=dilation_list,
            padding=padding,
            use_skip_connections=skips,
            dropout_rate=dropout,
            activation=activation,
            kernel_initializer=init,
            use_batch_norm=False,
            use_layer_norm=True,
            return_sequences=False,
            nb_stacks=num_stacks
            )(i_x)
        tcn_compl.append(o_tcn)
    # Concatenate and flatten different convolutional
    # Filter outputs
    if len(filter_sizes) > 1:
        tcn_compl = Concatenate()(tcn_compl)
        o_x = Dropout(dropout)(tcn_compl)
    else:
        o_x = Dropout(dropout)(tcn_compl[0])
    # o_x = Flatten()(o_x)
    model_x = Model(inputs=i_x, outputs=o_x)

    # input layer for author embeddings:
    i_author = Input(shape=input_shape_author)
    logger.info(f"input shape author embs: (author_emb_dims): {input_shape_author}")
    # Input layer for topic embeddings:
    i_topic = Input(shape=input_shape_topic)
    logger.info(f"input shape topic embs: (topic_emb_dims): {input_shape_topic}")
    # Combine these inputs
    o_comb = Concatenate()([i_author, i_topic])

    # Create small submodel
    model_other = Model(
        inputs=[i_author, i_topic],
        outputs=o_comb)

    # Combine feature vectors
    features_comb = Concatenate()([model_x.output, model_other.output])

    # Dense completely connected layer
    dense = Dense(
        units=dense_layers,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=l2(l2_float))(features_comb)
    dense = Dropout(dropout)(dense)

    # Output layer
    o = Dense(
        units=1,
        activation="sigmoid"
    )(dense)

    # Define model
    model_tot = Model(inputs=[model_x.input, model_other.input], outputs=[o])

    # Compile model
    model_tot.compile(
        loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return(model_tot, param_dict)

# check_model = tcn1((-1, 300), logger=get_logger("temporal_check"))
# check_model.summary()

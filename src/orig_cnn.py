from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, InputLayer, Conv1D, GlobalMaxPool1D, Concatenate, Flatten, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.regularizers import l2

from logger import get_logger


def cnn1(input_shape, logger, dense_layers=16, optimizer="Adam", 
         init="he_normal",
         num_filters=128, filter_sizes=(3, 4, 5), activation="relu",
         padding="valid"):
    """Convolution Network without merging of other feature vectors
    """

    # Check if the right shape was passed as argument
    if len(input_shape) != 2:
        raise Exception("Input shape needs to have 2 dimensions.",
                        "input shape:", input_shape)

    logger.info(f"input shape: {input_shape}")
    # Input layer
    i = Input(shape=input_shape)  # (words/sent, word_emb)
    # embedding_size = input_shape[1]

    # Output layer
    logger.info(f"Filter sizes: {filter_sizes}")
    conv_compl = []
    for filter_size in filter_sizes:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=(
                filter_size  # ,  # Word scope
                # embedding_size  # word embedding dimensions
                ),
            strides=1,
            padding=padding,
            activation=activation,
            kernel_initializer=init,
            data_format="channels_last"
            )(i)
        pooled = GlobalMaxPool1D()(conv)
        conv_compl.append(pooled)

    # Concatenate and flatten different convolutional
    # Filter outputs
    conv_merged = Concatenate()(conv_compl)
    flat = Flatten()(conv_merged)

    # Dense completely connected layer
    dense = Dense(
        units=dense_layers,
        activation=activation,
        kernel_initializer=init)(flat)

    # Output layer
    o = Dense(
        units=1,
        activation="sigmoid"
    )(dense)

    # Define model
    model = Model(inputs=[i], outputs=[o])

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return(model)


def cnn2(input_shape_x, input_shape_author, input_shape_topic,
         logger, dense_layers=32, optimizer="Adam",
         init="he_normal",
         num_filters=32, filter_sizes=(3, 4, 5), activation="relu",
         padding="valid", l2_float=0.0, dropout=0.0):
    """Convolution Network WITH merging of other feature vectors
    """

    param_dict = {
        "num filters": num_filters,
        "filter sizes": filter_sizes,
        "padding": padding,
        "activation": activation,
        "n dense layer": dense_layers,
        "l2 regu": l2_float,
        "drop-out": dropout,
        "kernel init": init
    }
    # Check if the right shape was passed as argument
    if len(input_shape_x) != 2:
        raise Exception("Input shape needs to have 2 dimensions.",
                        "input shape:", input_shape_x)

    logger.info(f"input shape post (word/sent, word_emb_dims): {input_shape_x}")

    # Input layer
    i_x = Input(shape=input_shape_x)  # (words/sent, word_emb)
    # embedding_size = input_shape[1]

    # Convolutional layer layer
    logger.info(f"Filter sizes: {filter_sizes}")
    conv_compl = []
    for filter_size in filter_sizes:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=(
                filter_size  # ,  # Word scope
                # embedding_size  # word embedding dimensions
                ),
            strides=1,
            padding=padding,
            activation=activation,
            kernel_initializer=init,
            data_format="channels_last",
            kernel_regularizer=l2(l2_float)
            )(i_x)
        pooled = GlobalMaxPool1D()(conv)
        conv_compl.append(pooled)

    # Concatenate and flatten different convolutional
    # Filter outputs
    conv_merged = Concatenate()(conv_compl)
    conv_merged = Dropout(dropout)(conv_merged)
    o_x = Flatten()(conv_merged)
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


# # Simple comment only model
# check_model = cnn1(input_shape=(100, 300), logger=get_logger("test"))

# # Complicated model with context
# check_model = cnn2(
#     input_shape_x=(100, 300),
#     input_shape_author=(int(100),),
#     input_shape_topic=(int(100),),
#     logger=get_logger("cnn_test"),
#     dense_layers=64
#     )

# check_model.summary()

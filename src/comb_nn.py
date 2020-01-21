from tcn import TCN

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Concatenate, Dropout, Conv1D, GlobalMaxPool1D
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l2

from logger import get_logger


def comb_nn(input_shape_x, input_shape_author, input_shape_topic, logger, 
            dense_layers=32, learn_rate=0.001, activation="relu", init="he_normal",
            num_filters_tcn=32, filter_sizes_tcn=[3], skips=True, num_stacks=1,
            num_filters_cnn=32, filter_sizes_cnn=[3, 4, 5],
            padding="same", l2_float=0.001, dropout=0.2,
         ):
    """Convolution Network and Temporal Convolutional network WITH merging of other feature vectors
    """
    optimizer = Adam(learning_rate=learn_rate)

    param_dict = {
        "padding": padding,
        "activation": activation,
        "n dense layer": dense_layers,
        "l2 regu": l2_float,
        "drop-out": dropout,
        "kernel init": init,
        "learn rate": learn_rate,
        "num filters TCN": num_filters_tcn,
        "filter sizes TCN": filter_sizes_tcn,
        "num stacks": num_stacks,
        "skip connections": skips,
        "num filters CNN": num_filters_cnn,
        "filter sizes CNN": filter_sizes_cnn}

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

    #  ========= Convolutional layes ==========================

    # -------------- TCN --------------------------
    logger.info(f"Filter sizes: {filter_sizes_tcn}")
    tcn_compl = []
    for filter_size in filter_sizes_tcn:

        # Determine dilation list:
        dilation_list = calc_dilations(
            filter_size=filter_size,
            field=input_shape_x[0],
            stacks=num_stacks)
        logger.info(f"Dilation list: {dilation_list}")

        single_tcn = TCN(
            nb_filters=num_filters_tcn,
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
        tcn_compl.append(single_tcn)
    # Concatenate and flatten different convolutional
    # Filter outputs
    if len(filter_sizes_tcn) > 1:
        o_tcn = Concatenate()(tcn_compl)
        o_tcn = Dropout(dropout)(o_tcn)
    else:
        o_tcn = Dropout(dropout)(tcn_compl[0])
    
    # ---------------- CNN -----------------------------
    logger.info(f"Filter sizes cnn: {filter_sizes_cnn}")
    cnn_compl = []
    for filter_size in filter_sizes_cnn:
        single_cnn = Conv1D(
            filters=num_filters_cnn,
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
        pooled = GlobalMaxPool1D()(single_cnn)
        cnn_compl.append(pooled)

    # Concatenate and flatten different convolutional
    # Filter outputs
    o_cnn = Concatenate()(cnn_compl)
    o_cnn = Dropout(dropout)(o_cnn)
    o_cnn = Flatten()(o_cnn)

    # ------------ Combine convolutional layers -------------------------

    o_x = Concatenate()([o_tcn, o_cnn])
    model_x = Model(inputs=i_x, outputs=o_x)

    # =============== OTHER FEATURE VECTORS =================================

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

    # ======================== Dense layer =====================================
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

    # ========================= COMPILING ===============================

    # Define model
    model_tot = Model(inputs=[model_x.input, model_other.input], outputs=[o])

    # Compile model
    model_tot.compile(
        loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return(model_tot, param_dict)

# check_model = comb_nn((100, 300), logger=get_logger("temporal_check"))
# check_model.summary()

# # Simple comment only model
# check_model = cnn1(input_shape=(100, 300), logger=get_logger("test"))


# Complicated model with context
check_model, _ = comb_nn(
    input_shape_x=(100, 300),
    input_shape_author=(int(100),),
    input_shape_topic=(int(100),),
    logger=get_logger("cnn_comb_test"),
    dense_layers=64
    )

check_model.summary()

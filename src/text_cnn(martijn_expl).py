import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
        self, sequence_length, num_classes, vocab_size, word2vec_W, word_idx_map, user_embeddings, topic_embeddings,
        embedding_size, batch_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_author = tf.placeholder(tf.int32, [None], name="input_author")
        self.input_topic = tf.placeholder(tf.int32, [None], name="input_topic")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Word vector:
            self.W = tf.Variable(word2vec_W,name="W")
            # user embeddings variable
            self.user_W = tf.Variable(user_embeddings, name = 'user_W')
            # Topic embeddings variable
            self.topic_W = tf.Variable(topic_embeddings, name = 'topic_W')
            # Select the embeddings corresponding to the words specified in 'input_x' (sequence_length)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # Expand last dimension index of comment embeddings. Why?
            # Should now have the following shape:
            # [batch, in-height, in_width, in_channels]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.user_embedding_vectors = tf.nn.embedding_lookup(self.user_W, self.input_author)
            self.topic_embedding_vectors = tf.nn.embedding_lookup(self.topic_W, self.input_topic)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = (
                    [filter_size, # 1, 2, or 3 (words scope) 
                    embedding_size, # Capture all the word embedding's dimensions
                    1, # Only 1 channel as input is used
                    num_filters]) # number of output channels = num of features per filter patch
                # Weight:
                # Random initialization of weights from truncated normal distribution for filter
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # Bias:
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    # Input: comment's word embeddings in shape:
                    # [batch, in-height, in_width, in_channels] =
                    # [100, 1/2/3 (filter determined), embedding size, 1]
                    self.embedded_chars_expanded,
                    # The filters with randomized weights:
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # height = (sequence_length - filter size + 2*padding)/stride + 1 = (100 - [1,2,3] + 0) / 1 + 1
                # Output shape of 1 filter of size 1 per batch: [100, emb_size]
                # Output shape of 1 filter of size 2 per batch: [99, emb_size]
                # Output shape of 1 filter of size 3 per batch: [98, emb_size]

                # 128 filters per filter size, so: [1, 100, emb_size, 128]

                # Maxpooling over the outputs
                # input shape:
                # 
                pooled = tf.nn.max_pool(
                    h, # Tensor of rank N+2 ([batch_size] + input_spatial_shape + [num_channels])
                    # Results in a scalar per batch per filter size per filter. 
                    ksize=[1, sequence_length - filter_size + 1, 1, 1], # size of window for each dimension of the input tensor
                    strides=[1, 1, 1, 1], # Stride for each dimension of the input vector 
                    padding='VALID', # output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i])
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # concatenate over the filter dimension?
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total],name="h_pool_flat")

        # Add another layer
        with tf.name_scope("last_layer"):
            C = tf.Variable(tf.random_normal([num_filters_total,100]),name="C")
            b_prime = tf.Variable( tf.constant(0.1,shape=[100]),name="b_prime")
            self.h_last = tf.nn.xw_plus_b(self.h_pool_flat,C,b_prime, name="h_last")
            
        # Add user embeddings
        with tf.name_scope("user_embedding_layer"):
            U = tf.Variable(tf.random_normal([300,400]), name="U")
            b_user = tf.Variable( tf.constant(0.1, shape=[400], name='b_user'))
            self.combined_vectors = tf.concat([self.h_last,self.user_embedding_vectors], 1)
            self.combined_vectors = tf.concat([self.combined_vectors,self.topic_embedding_vectors], 1)
            self.final_vector = tf.nn.relu(tf.nn.xw_plus_b(self.combined_vectors, U, b_user), name = 'final_vector')
#         with tf.name_scope("another_layer"):
#             V = tf.Variable(tf.random_normal([100,300]), name = 
                         
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.final_vector, self.dropout_keep_prob, name="h_drop")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[400, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            print(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("correct_preds"):
            self.correct_predictions = correct_predictions
        
        with tf.name_scope("predictions"):
            self.predictions
        
        with tf.name_scope("confusion_matrix"):
            self.labels = tf.argmax(self.input_y, 1, name="labels")
            self.confusion_matrix = tf.contrib.metrics.confusion_matrix( labels=self.labels, predictions=self.predictions, num_classes = 2, name="confusion_matrix")
            
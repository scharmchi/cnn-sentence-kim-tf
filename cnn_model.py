import tensorflow as tf
import numpy as np


class SentenceCNN(object):
    """
    This is a CNN for sentence classification derived from the Kim's paper
    The architecture is like Embedding + Conv + Maxpool + Softmax
    """
    def __init__(self, seq_len, n_classes):
        self.x = tf.placeholder(tf.int32, [None, seq_len], name="X")
        self.y = tf.placeholder(tf.float32, [None, n_classes], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _embedding_layer(self, vocab_size, embedding_size):
        with tf.device('/cpu:0'), tf.name_scope("Embedding Layer"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_embed")
            self.embedded = tf.nn.embedding_lookup(W, self.x)
            # Now we have a tensor of shape [None, sequence_length, embedding_size]
            # conv2d expects 4-D tensor (batch, width, height and channel) so
            # add the last dim manually
            self.embedded_expanded = tf.expand_dims(self.embedded, -1)
            # Now the shape we have is [None, sequence_length, embedding_size, 1]

    # n_filters_per_size: We have 128 of each filter_size
    def _conv_maxpool_layer(self, filter_sizes, embedding_size, n_filters_per_size, seq_len):
        pooled_outs = []
        for _, fil_height in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % fil_height):
                # height, width, depth (channels), total_num_filters
                filter_shape = [fil_height, embedding_size, 1, n_filters_per_size]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
                b = tf.Variable(tf.constant(0.1, shape=[n_filters_per_size]), name="b_conv")
                conv = tf.nn.conv2d(self.embedded_expanded, W, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, seq_len - fil_height + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="maxpool")
                pooled_outs.append(pooled)
        total_num_filters = n_filters_per_size * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outs)
        # look at example in https://www.tensorflow.org/versions/master/api_docs/python/array_ops.html#concat for above
        # below Once we have all the pooled output tensors from each filter size we combine them
        # into one long feature vector of shape [batch_size, total_num_filters]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_num_filters])

        # add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def _fc_output_layer(self, total_num_filters, n_classes):
        with tf.name_scope("output"):
            W = tf.get_variable(name="W", shape=[total_num_filters, n_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="b")
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # having a softmax is not necessary cause anyway I am picking the highest score
            # softmax is just like normalizing
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores-unnormalized")

    def _loss(self, lambda_l2_reg, l2_loss):
        with tf.name_scope("loss"):
            losses_vec = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y)
            self.loss = tf.reduce_mean(losses_vec) + lambda_l2_reg * l2_loss

    def _accuracy(self):
        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            correct_pred = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")


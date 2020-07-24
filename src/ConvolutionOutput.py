import operator
from functools import reduce

import tensorflow as tf


class ConvolutionOutput(tf.keras.layers.Layer):
    def __init__(self, target_shape, num_layers, d_model, rate=0.1):
        super(ConvolutionOutput, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.conv_layers = [
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv3DTranspose(filters=10, kernel_size=1, strides=1, padding='same',
                                                activation='relu')),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv3DTranspose(filters=2, kernel_size=1, strides=1, activation='relu'))
        ]

        self.reshape = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Reshape(target_shape=target_shape))  # shape before flattend in the input convolution

        self.dense = tf.keras.layers.Dense(
            reduce(operator.mul, target_shape, 1))  # shape when flattend in the input convolution

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        x = self.dropout(x, training=training)

        x = self.dense(x, training=training)
        x = self.reshape(x, training=training)
        out1 = x
        for conv_layer in self.conv_layers:
            x = conv_layer(x, training=training)

        return out1 + x  # target_shape

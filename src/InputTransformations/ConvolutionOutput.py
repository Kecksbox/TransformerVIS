import tensorflow as tf

import operator
from functools import reduce

from src.Layer.GRUGate import GRUGate


class ConvolutionOutput(tf.keras.layers.Layer):
    def __init__(self, convolutions, voxel_shape, target_shape, dff_decoder, rate=0.1):
        super(ConvolutionOutput, self).__init__()

        self.conv_layers = []
        for conv in reversed(convolutions):
            self.conv_layers.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv3DTranspose(
                        filters=conv['filters'],
                        kernel_size=conv['kernel_size'],
                        strides=conv['strides'],
                        activation=None,
                    )
                )
            )

        self.conv_layers.append(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv3DTranspose(
                    filters=voxel_shape[-1],
                    kernel_size=1,
                    strides=1,
                    activation=None,
                )
            )
        )

        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())

        self.reshape = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Reshape(target_shape=target_shape))  # shape before flattend in the input convolution
        self.reshape_out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Reshape(target_shape=voxel_shape))  # shape when fed into the network

        self.dense_r = tf.keras.layers.Dense(reduce(operator.mul, voxel_shape, 1), activation=None)
        self.dense = tf.keras.layers.Dense(reduce(operator.mul, target_shape, 1), activation=None)

        self.gru = GRUGate(reduce(operator.mul, voxel_shape, 1))

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        residual = self.dense_r(x)

        convolution = self.dense(x, training=training)
        convolution = self.reshape(convolution, training=training)

        for conv_layer in self.conv_layers:
            convolution = conv_layer(convolution, training=training)

        out = self.reshape_out(residual + self.flatten(convolution), training=training)

        out = self.dropout(out, training=training)

        return out  # target_shape
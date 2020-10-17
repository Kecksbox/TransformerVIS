import tensorflow as tf

import operator
from functools import reduce

from src.Layer.GRUGate import GRUGate


class ConvolutionOutput(tf.keras.layers.Layer):
    def __init__(self, convolutions, voxel_shape, target_shape, dff_decoder, rate=0.1):
        super(ConvolutionOutput, self).__init__()

        self.reshape_out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Reshape(target_shape=voxel_shape))  # shape when fed into the network

        self.dense_fit = self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(reduce(operator.mul, voxel_shape, 1), activation=None)  # shape when flattend in the input convolution
        ])

    def call(self, x, training):

        out = self.reshape_out(self.dense_fit(x, training=training), training=training)

        return out  # target_shape

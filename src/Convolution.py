import math
import operator
from functools import reduce

import tensorflow as tf

from src.GRUGate import GRUGate


class Convolution(tf.keras.layers.Layer):
    def __init__(self, convolutions, d_model, rate=0.1):
        super(Convolution, self).__init__()

        self.flatten_r = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())

        self.dense_r = tf.keras.layers.Dense(d_model, activation=None)
        self.dense = tf.keras.layers.Dense(d_model, activation='relu')

        self.conv_layers = []
        for conv in convolutions:
            self.conv_layers.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv3D(
                        filters=conv[0],
                        kernel_size=conv[1],
                        strides=conv[2],
                        activation=conv[3],
                    )
                )
            )

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gru = GRUGate(d_model)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        residual = self.dense_r(self.flatten_r(x, training=training))

        inp = self.layernorm(x)

        convolution = inp
        for conv_layer in self.conv_layers:
            convolution = conv_layer(convolution, training=training)
        out = self.dense(self.flatten(convolution, training=training), training=training)

        out = self.gru(residual, out)
        out = self.dropout(out, training=training)

        return out  # (batch_size, input_seq_len, d_model)

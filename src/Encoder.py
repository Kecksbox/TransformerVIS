import math

import numpy as np
import tensorflow as tf

from src.EncoderLayer import EncoderLayer


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, d_latent, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = []
        d_diff = d_model - d_latent
        d_diff_step = math.floor(d_diff / num_layers)
        for i in range(num_layers):
            calc_heads = math.floor(d_model / self.d_model * num_heads)
            if i == num_layers - 1:
                self.enc_layers.append(EncoderLayer(d_model, d_latent, calc_heads, dff, rate))
            else:
                self.enc_layers.append(EncoderLayer(d_model, d_model - d_diff_step, calc_heads, dff, rate))
            d_model -= d_diff_step

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding position encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

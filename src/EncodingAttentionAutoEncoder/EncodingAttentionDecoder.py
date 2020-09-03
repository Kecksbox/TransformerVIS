import numpy as np
import tensorflow as tf

from src.EncodingAttentionAutoEncoder.EncodingAttentionDecoderLayer import EncodingAttentionDecoderLayer


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


class EncodingAttentionDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(EncodingAttentionDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.unfold = tf.keras.layers.Dense(d_model)

        self.dec_layers = [EncodingAttentionDecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, padding_mask):

        x = self.dropout(x, training=training)

        x = self.unfold(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training, padding_mask)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x

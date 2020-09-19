import numpy as np
import tensorflow as tf

from src.EncodingAttentionAutoEncoder.EncodingAttentionDecoderLayer import EncodingAttentionDecoderLayer


class EncodingAttentionDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, dff, rate=0.1):
        super(EncodingAttentionDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.unfold = tf.keras.layers.Dense(d_model)

        self.dec_layers = [EncodingAttentionDecoderLayer(d_model, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, padding_mask):

        x = self.dropout(x, training=training)

        x = self.unfold(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training, padding_mask)

        return x

import tensorflow as tf

from src.MultiHeadAttention import MultiHeadAttention
from src.PointWiseFeedForward import PointWiseFeedForward


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_inp, d_tar, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_inp, num_heads)
        self.ffn_r = tf.keras.layers.Dense(d_tar, activation=None)
        self.ffn = PointWiseFeedForward(d_tar, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_r_output = self.ffn_r(out1)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_r_output + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

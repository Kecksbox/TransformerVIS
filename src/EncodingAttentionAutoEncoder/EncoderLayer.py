import tensorflow as tf

from src.MultiHeadAttention import MultiHeadAttention
from src.PointWiseFeedForward import PointWiseFeedForward
from src.GRUGate import GRUGate


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_inp, num_heads, dff, d_tar, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_inp, num_heads)
        self.ffn_r = tf.keras.layers.Dense(d_tar, activation=None)
        self.ffn = PointWiseFeedForward(d_tar, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gru1 = GRUGate(d_inp)
        self.gru2 = GRUGate(d_tar)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # attn_output, _ = self.mha(in1, in1, in1, mask)  # (batch_size, input_seq_len, d_model)
        #attn_output = tf.keras.activations.relu(attn_output, alpha=0.0, max_value=None, threshold=0)
        #out1 = self.gru1(x, attn_output, training=training)  # (batch_size, input_seq_len, d_model)
        #out1 = self.dropout1(out1, training=training)

        in2 = self.layernorm1(x)
        ffn_r_output = self.ffn_r(in2)
        ffn_output = self.ffn(in2)  # (batch_size, input_seq_len, d_model)
        ffn_output = tf.keras.activations.relu(ffn_output, alpha=0.0, max_value=None, threshold=0)
        out2 = self.gru2(ffn_r_output, ffn_output)  # (batch_size, input_seq_len, d_model)
        out2 = self.dropout2(out2, training=training)

        return out2

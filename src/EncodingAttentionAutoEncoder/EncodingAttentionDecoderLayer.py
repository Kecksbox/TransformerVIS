import tensorflow as tf

from src.MultiHeadAttention import MultiHeadAttention
from src.PointWiseFeedForward import PointWiseFeedForward
from src.GRUGate import GRUGate


class EncodingAttentionDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncodingAttentionDecoderLayer, self).__init__()

        # self.mha1 = MultiHeadAttention(d_model, num_heads)

        self.ffn = PointWiseFeedForward(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gru1 = GRUGate(d_model)
        self.gru2 = GRUGate(d_model)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        """
        in1 = x
        in1 = self.layernorm1(in1)
        attn1, attn_weights_block1 = self.mha1(
            enc_output, enc_output, in1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = tf.keras.activations.relu(attn1, alpha=0.0, max_value=None, threshold=0)
        out1 = self.gru1(x, attn1)  # (batch_size, target_seq_len, d_model)
        out1 = self.dropout1(out1, training=training)
        """

        in2 = self.layernorm2(x)
        ffn_output = self.ffn(in2)  # (batch_size, target_seq_len, d_model)
        ffn_output = tf.keras.activations.relu(ffn_output, alpha=0.0, max_value=None, threshold=0)
        out2 = self.gru2(x, ffn_output)  # (batch_size, target_seq_len, d_model)
        out2 = self.dropout2(out2, training=training)

        return out2

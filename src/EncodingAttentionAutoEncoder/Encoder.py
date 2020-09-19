import tensorflow as tf

from src.Utilities.PositionalEncoding import positional_encoding
from src.EncodingAttentionAutoEncoder.EncoderLayer import EncoderLayer
from src.Layer.SingleHeadAttention import SingleHeadAttention
from src.Layer.GRUGate import GRUGate


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, encoder_specs, num_attention_layers, att_dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model

        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.sha = SingleHeadAttention(d_model, att_dff, num_attention_layers, rate)

        self.layernorm_att = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gru_att = GRUGate(d_model)

        self.dropout_enc = tf.keras.layers.Dropout(rate)
        self.dropout_att = tf.keras.layers.Dropout(rate)

        self.num_layers = encoder_specs.__len__()
        self.enc_layers = [None] * self.num_layers
        for i in range(self.num_layers):
            spec = encoder_specs[i]
            inp = d_model
            if i != 0:
                inp = encoder_specs[i - 1][1]
            self.enc_layers[i] = EncoderLayer(224, spec[1], rate)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # adding position encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        in1 = self.layernorm_att(x)
        attn, attention_weights['selfattention_decoder_layer_block'] = self.sha(in1, in1, in1, mask)
        attn = tf.keras.activations.relu(attn, alpha=0.0, max_value=None, threshold=0)
        out1 = self.gru_att(x, attn)  # (batch_size, input_seq_len, d_model)
        out1 = self.dropout_att(out1, training=training)

        for i in range(self.num_layers):
            out1 = self.enc_layers[i](out1, training)

        return out1, attention_weights  # (batch_size, input_seq_len, d_model)

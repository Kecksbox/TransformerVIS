import tensorflow as tf

from src.Convolution import Convolution
from src.ConvolutionOutput import ConvolutionOutput
from src.Decoder import Decoder
from src.Encoder import Encoder
from src.Masks import create_masks


class Transformer(tf.keras.Model):
    def __init__(self, voxel_shape, d_model, d_latent, num_layers, num_heads, dff, max_length, SOS, EOS, rate=0.1):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.SOS = SOS
        self.EOS = EOS

        self.input_convolution = Convolution(voxel_shape, 2, d_model, rate)

        self.output_convolution = ConvolutionOutput(voxel_shape, 2, d_model, rate)

        self.encoder = Encoder(num_layers, d_model, d_latent, num_heads, dff, max_length, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, max_length, rate)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        inp_embedding = self.input_convolution(inp, training=training)  # (batch_size, inp_seq_len, d_model)

        enc_output = self.encoder(inp_embedding, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        tar_embedding = self.input_convolution(tar, training=training)  # (batch_size, inp_seq_len - 1, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar_embedding, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.output_convolution(dec_output)  # (batch_size, tar_seq_len, ...voxel_shape)

        return final_output, enc_output, attention_weights

    @tf.function
    def evaluate(self, input):
        # adding the start and end token
        input = tf.concat([[self.SOS], input, [self.EOS]], 0)

        inp_sentence = input
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # the first voxel to the transformer should be the
        # start voxel.
        decoder_input = tf.expand_dims(tf.expand_dims(self.SOS, 0), 0)
        output = tf.cast(decoder_input, tf.float32)

        for i in range(self.max_length - 1):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, ...voxel_shape)
            predictions, latent, attention_weights = self(encoder_input,
                                                          output,
                                                          False,
                                                          enc_padding_mask,
                                                          combined_mask,
                                                          dec_padding_mask)

            # select the last voxel from the seq_len dimension
            prediction = predictions[:, -1:, :]  # (batch_size, 1, ...voxel_shape)

            # return the result if the prediction is equal to the end token [TODO]
            # if all(tf.math.equal(prediction[0][0], EOS)):
            #    return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_voxel to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, prediction], axis=1)

        return tf.squeeze(output, axis=0), latent, attention_weights

import copy
import math
import operator
from functools import reduce

import tensorflow as tf

from src.Convolution import Convolution
from src.ConvolutionOutput import ConvolutionOutput
from src.Decoder import Decoder
from src.Encoder import Encoder
from src.Masks import create_masks


class Transformer(tf.keras.Model):
    def __init__(self, voxel_shape, d_model,
                 num_convolutions,
                 # ([index: (d_inp, num_heads, dff, d_tar)])
                 encoder_specs, num_heads,
                 num_layers_decoder, dff_decoder,
                 max_length, SOS, EOS, PAD_TOKEN,
                 convolution_scaling=2,
                 rate=0.1):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.SOS = SOS
        self.EOS = EOS
        self.PAD_TOKEN = PAD_TOKEN

        self.create_convolutions(num_convolutions, voxel_shape, d_model, dff_decoder, convolution_scaling, rate)

        self.encoder = Encoder(d_model, encoder_specs, num_heads, max_length, rate)

        self.decoder = Decoder(num_layers_decoder, d_model, num_heads, dff_decoder, max_length, rate)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        inp_embedding = self.input_convolution(inp, training=training)  # (batch_size, inp_seq_len, d_model)

        enc_output = self.encoder(inp_embedding, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_latent)

        tar_embedding = self.input_convolution(tar, training=training)  # (batch_size, inp_seq_len - 1, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar_embedding, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.output_convolution(dec_output)  # (batch_size, tar_seq_len, ...voxel_shape)

        return final_output, enc_output, attention_weights

    # @tf.function
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
                encoder_input, output, self.PAD_TOKEN)

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

    def compute_convolutions(self, num_convolutions, voxel_shape, d_model, convolution_scaling):
        shape = list(copy.deepcopy(voxel_shape))

        dim_channel = shape[-1]
        dim_max = max(shape)
        dim_reduction_per_step = (dim_max - 1) / num_convolutions

        dim_reduction_adjusted = math.floor(dim_reduction_per_step)
        dim_reduction_error = dim_reduction_per_step - dim_reduction_adjusted

        convolutions = []
        error = dim_reduction_error
        for i in range(num_convolutions):

            local_dim_reduction = dim_reduction_adjusted

            if error >= 1:
                error -= 1
                local_dim_reduction += 1
            if i == num_convolutions - 1:
                if error > 0:
                    local_dim_reduction += 1

            prev_dim = reduce(operator.mul, shape, 1)

            adj = [local_dim_reduction + 1] * 3
            for i in range(3):
                if (shape[i] <= adj[i]):
                    adj[i] = shape[i]
                shape[i] += 1 - adj[i]

            shape[-1] = dim_channel
            post_dim = reduce(operator.mul, shape, 1)
            dims_scale = prev_dim / post_dim
            dim_channel = max(int(dim_channel * dims_scale), 1)
            shape[-1] = dim_channel
            convolutions.append(
                [dim_channel, tuple(adj), 1, None]
            )
            error += dim_reduction_error

        scale_model = d_model / dim_channel
        for i in range(num_convolutions):
            adj_channels = max(int(convolutions[i][0] * scale_model), 1) * convolution_scaling
            convolutions[i][0] = adj_channels
            shape[-1] = adj_channels

        return [[(voxel_shape[-1], 1, 1, None)] + convolutions, shape]

    def create_convolutions(self, num_convolutions, voxel_shape, d_model, dff_decoder, convolution_scaling, rate):

        convolution = self.compute_convolutions(num_convolutions, voxel_shape, d_model, convolution_scaling)

        self.input_convolution = Convolution(convolution[0], d_model, rate)

        self.output_convolution = ConvolutionOutput(reversed(convolution[0]), convolution[1], voxel_shape, dff_decoder, rate)


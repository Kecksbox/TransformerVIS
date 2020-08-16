import datetime
import time

import tensorflow as tf

from src.Convolution import Convolution
from src.ConvolutionOutput import ConvolutionOutput
from src.Decoder import Decoder
from src.EncodingAttentionAutoEncoder.Encoder import Encoder
from src.Utilities.CustomSchedule import CustomSchedule
from src.Masks import create_padding_mask
from src.EncodingAttentionAutoEncoder.EncodingAttentionDecoder import EncodingAttentionDecoder
from src.Utilities.TokenBuilder import createFullToken


class EncodingAttentionAutoEncoder(tf.keras.Model):
    def __init__(self, voxel_shape, d_model,
                 seq_convolution,
                 # ([index: (d_inp, num_heads, dff, d_tar)])
                 encoder_specs,
                 num_heads,
                 num_layers_decoder, dff_decoder,
                 max_length, SOS, EOS, PAD_TOKEN,
                 beta_1=0.9, beta_2=0.98, epsilon=1e-9, warmup_steps=4000,
                 rate=0.1):
        super(EncodingAttentionAutoEncoder, self).__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.SOS = createFullToken(voxel_shape, SOS)
        self.EOS = createFullToken(voxel_shape, EOS)
        self.PAD_TOKEN = PAD_TOKEN

        self.input_convolution = Convolution(seq_convolution[0], d_model, rate)

        self.output_convolution = ConvolutionOutput(reversed(seq_convolution[0]), seq_convolution[1], voxel_shape, dff_decoder,
                                                    rate)

        self.encoder = Encoder(d_model, encoder_specs, num_heads, max_length, rate)

        self.decoder = EncodingAttentionDecoder(num_layers_decoder, d_model, num_heads, dff_decoder, max_length, rate)

        self.loss_object = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE, name='mean_squared_error'
        )
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.optimizer = tf.keras.optimizers.Adam(CustomSchedule(self.d_model, warmup_steps), beta_1=beta_1,
                                                  beta_2=beta_2, epsilon=epsilon)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/encodingAutoEncoder' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        self.load_latest_checkpoint()

    def load_latest_checkpoint(self):
        ckpt = tf.train.Checkpoint(selfattention=self, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(selfattention=self, optimizer=self.optimizer),
            "./checkpoints/encodingattention/train",
            max_to_keep=5
        )
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored!!')

    def call(self, inp, tar, training, enc_padding_mask, dec_padding_mask):
        inp_embedding = self.input_convolution(inp, training=training)  # (batch_size, inp_seq_len, d_model)

        enc_output = self.encoder(inp_embedding, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_latent)

        tar_embedding = self.input_convolution(tar, training=training)  # (batch_size, inp_seq_len - 1, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar_embedding, enc_output, training, dec_padding_mask)

        final_output = self.output_convolution(dec_output)  # (batch_size, tar_seq_len, ...voxel_shape)

        return final_output, enc_output, attention_weights

    @tf.function
    def train_step(self, inp):
        tar_inp = inp[:, :-1]
        tar_real = inp[:, 1:]

        enc_padding_mask = create_padding_mask(inp, self.PAD_TOKEN)
        dec_padding_mask = enc_padding_mask

        with tf.GradientTape() as tape:
            predictions, _, _ = self(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     dec_padding_mask)
            self.update(tar_real, predictions, tape)

    #@tf.function
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

            enc_padding_mask = create_padding_mask(encoder_input, self.PAD_TOKEN)
            dec_padding_mask = enc_padding_mask

            # predictions.shape == (batch_size, seq_len, ...voxel_shape)
            predictions, latent, attention_weights = self(encoder_input,
                                                          output,
                                                          False,
                                                          enc_padding_mask,
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

    def train(self, train_dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()

            for (batch, (parameters, run)) in enumerate(train_dataset):
                self.train_step(run)

            if (epoch + 1) % 20 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, self.train_loss.result()))

                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=epoch)

    def update(self, tar_real, predictions, tape):
        loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.reduce_all(tf.math.equal(real, self.PAD_TOKEN), [5, 4, 3, 2]))

        loss_ = tf.reduce_sum(self.loss_object(real, pred), [4, 3, 2])

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
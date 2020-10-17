import datetime
import time

import tensorflow as tf

from src.InputTransformations.Convolution import Convolution
from src.InputTransformations.ConvolutionOutput import ConvolutionOutput
from src.EncodingAttentionAutoEncoder.Encoder import Encoder
from src.Utilities.CustomSchedule import CustomSchedule
from src.Utilities.Masks import create_padding_mask
from src.EncodingAttentionAutoEncoder.EncodingAttentionDecoder import EncodingAttentionDecoder
from src.Utilities.TokenBuilder import createFullToken


class EncodingAttentionAutoEncoder(tf.keras.Model):
    def __init__(self, voxel_shape, d_model,
                 seq_convolution,
                 # ([index: (d_inp, num_heads, dff, d_tar)])
                 encoder_specs,
                 num_attention_layers,
                 att_dff,
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

        self.input_convolution = Convolution(seq_convolution, d_model, rate)

        self.output_convolution = ConvolutionOutput(seq_convolution,
                                                    voxel_shape,
                                                    self.input_convolution.getTargetShape(voxel_shape),
                                                    dff_decoder,
                                                    rate)

        self.encoder = Encoder(d_model, encoder_specs, num_attention_layers, att_dff, max_length, rate)

        self.decoder = EncodingAttentionDecoder(num_layers_decoder, d_model, dff_decoder, rate)

        self.loss_object = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
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

    def call(self, inp, training, enc_padding_mask, dec_padding_mask):
        inp_embedding = self.input_convolution(inp, training=training)

        enc_output, attention_weights = self.encoder(inp_embedding, training,
                                                     enc_padding_mask)

        dec_output = self.decoder(enc_output, training, dec_padding_mask)

        final_output = self.output_convolution(dec_output)

        return final_output, enc_output, attention_weights

    @tf.function
    def train_step(self, inp):
        tar_inp = inp[:, 1:-1]
        tar_real = inp[:, 1:-1]

        enc_padding_mask = create_padding_mask(tar_inp, self.PAD_TOKEN)
        dec_padding_mask = enc_padding_mask

        with tf.GradientTape() as tape:
            predictions, enc_output, _ = self(tar_inp,
                                     True,
                                     enc_padding_mask,
                                     dec_padding_mask)
            self.update(tar_real, predictions, enc_output, tape)

    def evaluate(self, input):

        inp_sentence = input
        encoder_input = tf.expand_dims(inp_sentence, 0)

        enc_padding_mask = create_padding_mask(encoder_input, self.PAD_TOKEN)
        dec_padding_mask = enc_padding_mask

        predictions, latent, attention_weights = self(encoder_input,
                                                      False,
                                                      enc_padding_mask,
                                                      dec_padding_mask)

        return tf.squeeze(predictions, axis=0), latent, attention_weights

    def train(self, train_dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()

            for (batch, (parameters, run)) in enumerate(train_dataset):
                self.train_step(run)

            if (epoch + 1) % 100 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, self.train_loss.result()))

                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=epoch)

    def update(self, tar_real, predictions, enc_output, tape):
        loss = self.loss_function(tar_real, predictions, enc_output)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def loss_function(self, real, pred, enc_output):
        mask = tf.math.logical_not(tf.reduce_all(tf.math.equal(real, self.PAD_TOKEN), [5, 4, 3, 2]))

        loss_ = tf.reduce_sum(self.loss_object(real, pred), [4, 3, 2])
        loss_ += tf.reduce_sum(tf.math.abs(enc_output), axis=-1)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss = tf.reduce_sum(loss_) / tf.reduce_sum(mask)

        self.train_loss(loss)

        return loss

import time

import tensorflow as tf

from src.Masks import create_masks

import datetime

from src.Utilities.CustomSchedule import CustomSchedule


class Trainer:
    def __init__(self, transformer, PAD_TOKEN=0, beta_1=0.9, beta_2=0.98, epsilon=1e-9, warmup_steps=4000):
        self.transformer = transformer
        self.PAD_TOKEN = PAD_TOKEN
        self.learningRate = CustomSchedule(transformer.d_model, warmup_steps)
        self.loss_object = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE, name='mean_squared_error'
        )
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.optimizer = tf.keras.optimizers.Adam(self.learningRate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        self.load_latest_checkpoint()

    def load_latest_checkpoint(self):
        ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer),
            "./checkpoints/vanillaTransformer/train",
            max_to_keep=5
        )
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored!!')

    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp, self.PAD_TOKEN)

        with tf.GradientTape() as tape:
            predictions, _, _ = self.transformer(inp, tar_inp,
                                                 True,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
            self.update(tar_real, predictions, tape)

    def train(self, train_dataset, epochs):

        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()

            for (batch, (paramters, inp)) in enumerate(train_dataset):
                self.train_step(inp, inp)

                """
                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                """

            if (epoch + 1) % 20 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, self.train_loss.result()))

                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
            """
            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                train_loss.result(),
                                                                train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
            """

    def update(self, tar_real, predictions, tape):
        loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.reduce_all(tf.math.equal(real, self.PAD_TOKEN), [5, 4, 3, 2]))

        loss_ = tf.reduce_sum(self.loss_object(real, pred), [4, 3, 2])

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)



current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/vaniallaTransformer' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

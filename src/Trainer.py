import time

import tensorflow as tf

from src.Masks import create_masks

from src.Utilities.GameOfLife import createTestSet, show


class Trainer:
    def __init__(self, transformer, PAD_TOKEN=0, beta_1=0.9, beta_2=0.98, epsilon=1e-9, warmup_steps=4000):
        self.transformer = transformer
        self.PAD_TOKEN = PAD_TOKEN
        self.learningRate = CustomSchedule(transformer.d_model, warmup_steps)
        self.optimizer = tf.keras.optimizers.Adam(self.learningRate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        self.load_latest_checkpoint()

    def load_latest_checkpoint(self):
        ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer),
            "./checkpoints/train",
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

            train_loss.reset_states()
            train_accuracy.reset_states()

            for (batch, (inp)) in enumerate(train_dataset):
                self.train_step(inp, inp)

                """
                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                """

            if (epoch + 1) % 200 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))
                print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                    train_loss.result(),
                                                                    train_accuracy.result()))

                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
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

        train_loss(loss)
        train_accuracy.update_state(tar_real, predictions)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.reduce_all(tf.math.equal(real, self.PAD_TOKEN), 5))

        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error')

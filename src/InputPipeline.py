import tensorflow as tf


class InputPipeline:

    def __init__(self, BUFFER_SIZE, BATCH_SIZE, shape, SOS, EOS, PAD_TOKEN):
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.shape = shape
        self.SOS = SOS
        self.EOS = EOS
        self.PAD_TOKEN = PAD_TOKEN

    def pre_process_internal(self, inp):
        return tf.concat([[self.SOS], inp, [self.EOS]], 0)

    def process(self, dataset):
        train_examples = dataset.map(self.pre_process_internal, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_examples.cache()
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE).padded_batch(
            self.BATCH_SIZE,
            padded_shapes=self.shape,
            padding_values=tf.constant(self.PAD_TOKEN, tf.float32),
        )
        return train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
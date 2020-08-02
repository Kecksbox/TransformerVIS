import tensorflow as tf


class InputPipeline:

    def __init__(self, shape, max_length, SOS, EOS, PAD_TOKEN, BUFFER_SIZE = 20000, BATCH_SIZE=64):
        self.shape = shape
        self.max_length = max_length
        self.SOS = SOS
        self.EOS = EOS
        self.PAD_TOKEN = PAD_TOKEN
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE

    def pre_process_internal(self, inp):
        return tf.concat([[self.SOS], inp, [self.EOS]], 0)

    def filter_max_length(self, x):
        return tf.size(x) <= self.max_length

    def process(self, dataset):
        train_examples = dataset.cache()
        train_examples = train_examples.map(self.pre_process_internal, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_examples.filter(self.filter_max_length)
        train_dataset = train_examples.cache()
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE).padded_batch(
            self.BATCH_SIZE,
            padded_shapes=self.shape,
            padding_values=tf.constant(self.PAD_TOKEN, tf.float32),
        )
        return train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

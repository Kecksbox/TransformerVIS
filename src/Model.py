import tensorflow as tf

from src.InputPipeline import InputPipeline
from src.Trainer import Trainer
from src.Transformer import Transformer


def createFullToken(voxel_shape, TOKEN):
    return tf.fill(voxel_shape, TOKEN)


class Model:

    def __init__(self, d_model,
                 num_convolutions,
                 # ([index: (d_inp, num_heads, dff, d_tar)])
                 encoder_specs,
                 num_heads,
                 num_layers_decoder, dff_decoder,
                 shape, max_length, SOS_TOKEN=-1, EOS_TOKEN=-2, PAD_TOKEN=0,
                 BUFFER_SIZE=20000, BATCH_SIZE=64,
                 dropout_rate=0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9, warmup_steps=4000,
                 ):
        voxel_shape = shape[1:]
        self.SOS = createFullToken(voxel_shape, SOS_TOKEN)
        self.EOS = createFullToken(voxel_shape, EOS_TOKEN)
        self.input_pipeline = InputPipeline(
            BUFFER_SIZE=BUFFER_SIZE,
            BATCH_SIZE=BATCH_SIZE,
            shape=shape,
            max_length=max_length,
            SOS=self.SOS,
            EOS=self.EOS,
            PAD_TOKEN=PAD_TOKEN,
        )
        self.trainer = Trainer(
            transformer=Transformer(
                voxel_shape=voxel_shape,
                d_model=d_model,
                num_convolutions=num_convolutions,
                encoder_specs=encoder_specs,
                num_heads=num_heads,
                num_layers_decoder=num_layers_decoder,
                dff_decoder=dff_decoder,
                max_length=max_length,
                SOS=self.SOS,
                EOS=self.EOS,
                PAD_TOKEN=PAD_TOKEN,
                rate=dropout_rate
            ),
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            warmup_steps=warmup_steps,
            PAD_TOKEN=PAD_TOKEN,
        )

    def train(self, set, epochs):
        self.trainer.train(
            train_dataset=set,
            epochs=epochs
        )
        return set

    def evaluate(self, inp):
        return self.trainer.transformer.evaluate(inp)

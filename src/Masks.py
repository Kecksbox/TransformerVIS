import tensorflow as tf


def create_padding_mask(seq):
    # reduce the last 4 dimension used by the voxels
    seq = tf.reduce_all(tf.reduce_all(tf.reduce_all(tf.reduce_all(tf.math.equal(seq, 0), 5), 4), 3), 2)

    # add extra dimensions to add the padding
    # to the attention logits.
    res = tf.cast(seq, tf.float32)
    return res[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    # It is the same a the Encoder padding mask in case of an autoencoder
    dec_padding_mask = enc_padding_mask

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

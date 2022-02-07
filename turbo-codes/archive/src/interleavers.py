import tensorflow as tf

def make_tf_interleaver(block_len):
    tf_interleaver = {}
    tf_interleaver["permutation"] = tf.random.shuffle(tf.range(block_len))
    tf_interleaver["depermutation"] = tf.math.invert_permutation(tf_interleaver["permutation"])
    return tf_interleaver
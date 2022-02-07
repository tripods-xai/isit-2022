import tensorflow as tf

def data_generator(batch_size, block_len, model):
    while True:
        input_bits = tf.cast(tf.random.uniform((batch_size, block_len), minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)
        yield (input_bits, input_bits)
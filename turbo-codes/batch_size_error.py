import tensorflow as tf
import numpy as np


def base_2_accumulator(length: int):
    powers_of_2 = tf.bitwise.left_shift(1, tf.range(length))
    return powers_of_2[::-1]

rng = np.random.default_rng(0)

dump_file = './cpu_outputs.txt'
# dump_file = './gpu_outputs.txt'

f = open(dump_file, 'w')

batch_size = 1000
block_len=33
msg = tf.constant(rng.integers(0, 2, size=(batch_size, block_len, 1)), dtype=tf.int32)
print('msg', file=f)
print(msg, file=f)

window = 5
base_2 = base_2_accumulator(window)
print('base_2', file=f)
print(base_2, file=f)
print(base_2)

base_2_filter = tf.cast(base_2[:, None, None], dtype=tf.float32)
print('base_2_filter', file=f)
print(base_2_filter, file=f)

msg_prepended = tf.pad(msg[:, :, 0], paddings=tf.constant([[0, 0], [window-1, 0]]))
print('msg_prepended', file=f)
print(msg_prepended, file=f)

conv_msg_input = tf.cast(msg_prepended[:,:,None], dtype=tf.float32)
print('conv_msg_input', file=f)
print(conv_msg_input, file=f)

state_sequence = tf.cast(tf.nn.conv1d(conv_msg_input, base_2_filter, stride=1, padding='VALID'), dtype=tf.int32)[:,:,0]
print('state_sequence', file=f)
print(state_sequence, file=f)
print('first binary sequence', file=f)
print(msg_prepended[0].numpy().tolist(), file=f)
print('first state sequence', file=f)
print(state_sequence[0].numpy().tolist(), file=f)
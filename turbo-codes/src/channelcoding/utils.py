import tensorflow as tf


def dec2bitarray(arr, num_bits:int, little_endian: bool=False):
    if little_endian:
        shift_arr = tf.range(num_bits)
    else:
        shift_arr = tf.range(num_bits)[::-1]
    return tf.bitwise.right_shift(tf.expand_dims(arr, -1), shift_arr) % 2

def enumerate_binary_inputs(window: int):
    return dec2bitarray(tf.range(2 ** window), window)

def safe_int(x: float) -> int:
    assert x.is_integer()
    return int(x)

def base_2_accumulator(length: int, little_endian: bool=False):
    powers_of_2 = tf.bitwise.left_shift(1, tf.range(length))
    if little_endian:
        return powers_of_2
    else:
        return powers_of_2[::-1]

def bitarray2dec(arr, little_endian=False, axis=-1):
    base_2 = base_2_accumulator(arr.shape[axis], little_endian=little_endian)
    return tf.tensordot(arr, base_2, axes=[[axis], [0]])

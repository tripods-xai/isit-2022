import tensorflow as tf

from src.channelcoding.interleavers import PermuteInterleaver

def test_permute_interleaver_basic():
    block_len = 100
    interleaver = PermuteInterleaver(block_len)
    assert interleaver.block_len == block_len
    assert interleaver.permutation.shape[0] == block_len
    assert interleaver.depermutation.shape[0] == block_len

    deinterleaver = interleaver.deinterleaver()

    assert tf.reduce_all(interleaver.permutation == deinterleaver.depermutation)
    assert tf.reduce_all(deinterleaver.permutation == interleaver.depermutation)

    # Could fail 1/(block_len!) times. Just rerun. Checking to make sure the other tests are not passing
    # because interleaver is just doing an identity interleaving
    msg = tf.stack([tf.range(block_len), -1 * tf.range(block_len)], axis=0)
    assert tf.reduce_any(interleaver(msg) != msg)
    assert tf.reduce_all(deinterleaver(interleaver(msg)) == msg)


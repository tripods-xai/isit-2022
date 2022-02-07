import tensorflow as tf
from src.channelcoding.channels import AWGN

import numpy as np
from numpy.testing import assert_array_almost_equal
import commpy.channelcoding as cc

from src.channelcoding.codes import IdentityCode
from src.channelcoding.encoders import AffineConvolutionalCode
from src.channelcoding.bcjr import BCJRDecoder, HazzysTurboDecoder, PriorInjector, TurboDecoder
from src.channelcoding.interleavers import PermuteInterleaver
from tests.channelcoding.utils import interleaver_to_commpy, vturbo_decode, vhazzys_turbo_decode

from .. import modified_convcode as mcc
from .. import modified_turbo as mt

def test_compare_tf_map_decode_to_commpy_map_decode_no_noise():
    gen = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias = tf.constant([0, 0])
    code = AffineConvolutionalCode(gen, bias)

    sigma = 1.
    channel = IdentityCode() * 2. - 1.

    prior = PriorInjector()
    decoder = prior.and_then(BCJRDecoder(code.trellis, AWGN(sigma), use_max=False))

    encoder_channel = code.and_then(channel)

    # Two messages of time 20 and 1 channel
    input_bits = tf.random.uniform((2, 20, 1), maxval=2, dtype=tf.int32)
    received_msg = encoder_channel(input_bits)
    tf_confidence = decoder(received_msg)

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    commpy_received = 2. * np.stack([
        cc.conv_encode(input_bits.numpy()[0, :, 0], commpy_trellis, termination='cont').reshape(20, 2),
        cc.conv_encode(input_bits.numpy()[1, :, 0], commpy_trellis, termination='cont').reshape(20, 2)],
        axis=0) - 1.
    np_received = received_msg.numpy()
    assert_array_almost_equal(np_received, commpy_received)

    L_int = np.zeros(input_bits.shape[1])
    L = np.stack([
        cc.map_decode(np_received[0, :, 0], np_received[0, :, 1], commpy_trellis, sigma ** 2, L_int, mode='compute')[0],
        cc.map_decode(np_received[1, :, 0], np_received[1, :, 1], commpy_trellis, sigma ** 2, L_int, mode='compute')[0]
    ], axis=0)[:, :, None]


    assert_array_almost_equal(L, tf_confidence.numpy(), decimal=5)

def test_compare_tf_map_decode_to_commpy_map_decode_with_noise():
    gen = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias = tf.constant([0, 0])
    code = AffineConvolutionalCode(gen, bias)

    sigma = 1.
    channel = AWGN(sigma)

    prior = PriorInjector()
    decoder = prior.and_then(BCJRDecoder(code.trellis, AWGN(sigma), use_max=False))

    encoder_channel = code.and_then(channel)

    # Two messages of time 20 and 1 channel
    input_bits = tf.random.uniform((2, 20, 1), maxval=2, dtype=tf.int32)
    received_msg = encoder_channel(input_bits)
    tf_confidence = decoder(received_msg)

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    np_received = received_msg.numpy()
    L_int = np.zeros(input_bits.shape[1])
    L = np.stack([
        cc.map_decode(np_received[0, :, 0], np_received[0, :, 1], commpy_trellis, sigma ** 2, L_int, mode='compute')[0],
        cc.map_decode(np_received[1, :, 0], np_received[1, :, 1], commpy_trellis, sigma ** 2, L_int, mode='compute')[0]
    ], axis=0)[:, :, None]

    assert_array_almost_equal(L, tf_confidence.numpy(), decimal=5)

def test_compare_tf_map_decode_to_commpy_map_decode_no_noise_nonzero_L_int():
    gen = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias = tf.constant([0, 0])
    code = AffineConvolutionalCode(gen, bias)

    sigma = 1.
    channel = IdentityCode() * 2. - 1.

    encoder_channel = code.and_then(channel)

    # Two messages of time 20 and 1 channel
    input_bits = tf.random.uniform((2, 20, 1), maxval=2, dtype=tf.int32)
    received_msg = encoder_channel(input_bits)

    L_int = tf.random.normal(input_bits.shape)[:, :, 0]
    prior = PriorInjector(L_int)
    decoder = prior.and_then(BCJRDecoder(code.trellis, AWGN(sigma), use_max=False))
    tf_confidence = decoder(received_msg)

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    np_received = received_msg.numpy()
    np_L_int = L_int.numpy()
    L = np.stack([
        cc.map_decode(np_received[0, :, 0], np_received[0, :, 1], commpy_trellis, sigma ** 2, np_L_int[0], mode='compute')[0],
        cc.map_decode(np_received[1, :, 0], np_received[1, :, 1], commpy_trellis, sigma ** 2, np_L_int[1], mode='compute')[0]
    ], axis=0)[:, :, None]

    assert_array_almost_equal(L, tf_confidence.numpy(), decimal=5)

def test_compare_tf_map_decode_to_commpy_map_decode_with_noise_nonzero_L_int():
    gen = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias = tf.constant([0, 0])
    code = AffineConvolutionalCode(gen, bias)

    sigma = 1.
    channel = AWGN(sigma)

    encoder_channel = code.and_then(channel)

    # Two messages of time 20 and 1 channel
    input_bits = tf.random.uniform((2, 20, 1), maxval=2, dtype=tf.int32)
    received_msg = encoder_channel(input_bits)

    L_int = tf.random.normal(input_bits.shape)[:, :, 0]
    prior = PriorInjector(L_int)
    decoder = prior.and_then(BCJRDecoder(code.trellis, AWGN(sigma), use_max=False))
    tf_confidence = decoder(received_msg)

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    np_received = received_msg.numpy()
    np_L_int = L_int.numpy()
    L = np.stack([
        cc.map_decode(np_received[0, :, 0], np_received[0, :, 1], commpy_trellis, sigma ** 2, np_L_int[0], mode='compute')[0],
        cc.map_decode(np_received[1, :, 0], np_received[1, :, 1], commpy_trellis, sigma ** 2, np_L_int[1], mode='compute')[0]
    ], axis=0)[:, :, None]

    assert_array_almost_equal(L, tf_confidence.numpy(), decimal=5)

def test_compare_tf_turbo_decode_to_commpy_turbo_decode_without_noise_one_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = IdentityCode() * 2. - 1.
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 1
    decoder = TurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    
    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)

def test_compare_tf_turbo_decode_to_commpy_turbo_decode_without_noise_two_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = IdentityCode() * 2. - 1.
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 2
    decoder = TurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)

   
def test_compare_tf_turbo_decode_to_commpy_turbo_decode_without_noise_six_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = IdentityCode() * 2. - 1.
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter=6
    decoder = TurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)


def test_compare_tf_turbo_decode_to_commpy_turbo_decode_with_noise_one_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = AWGN(sigma)
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 1
    decoder = TurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)


def test_compare_tf_turbo_decode_to_commpy_turbo_decode_with_noise_two_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = AWGN(sigma)
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 2
    decoder = TurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)

def test_compare_tf_turbo_decode_to_commpy_turbo_decode_with_noise_six_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = AWGN(sigma)
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 6
    decoder = TurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)

def test_hazzys_compare_tf_turbo_decode_to_commpy_turbo_decode_without_noise_one_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = IdentityCode() * 2. - 1.
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 1
    decoder = HazzysTurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vhazzys_turbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)

def test_hazzys_compare_tf_turbo_decode_to_commpy_turbo_decode_without_noise_two_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = IdentityCode() * 2. - 1.
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 2
    decoder = HazzysTurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vhazzys_turbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)

   
def test_hazzys_compare_tf_turbo_decode_to_commpy_turbo_decode_without_noise_six_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = IdentityCode() * 2. - 1.
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 6
    decoder = HazzysTurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vhazzys_turbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)


def test_hazzys_compare_tf_turbo_decode_to_commpy_turbo_decode_with_noise_one_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = AWGN(sigma)
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 1
    decoder = HazzysTurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vhazzys_turbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)


def test_hazzys_compare_tf_turbo_decode_to_commpy_turbo_decode_with_noise_two_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = AWGN(sigma)
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 2
    decoder = HazzysTurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vhazzys_turbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)

def test_hazzys_compare_tf_turbo_decode_to_commpy_turbo_decode_with_noise_six_iter():
    gen1 = tf.constant([[1, 1, 1], [1, 0 , 1]])
    bias1 = tf.constant([0, 0])
    code1 = AffineConvolutionalCode(gen1, bias1)

    gen2 = tf.constant([[1, 1, 1], [1, 0 , 0]])
    bias2 = tf.constant([0, 0])
    code2 = AffineConvolutionalCode(gen2, bias2)

    msg_length = 20
    batch_size = 2
    input_bits = tf.random.uniform((batch_size, msg_length, 1), maxval=2, dtype=tf.int32)

    interleaver = PermuteInterleaver(msg_length)

    turbo_encoder = code1.concat(interleaver.and_then(code2))
    sigma = 1.
    channel = AWGN(sigma)
    decoder1 = BCJRDecoder(code1.trellis, AWGN(sigma), use_max=False)
    decoder2 = BCJRDecoder(code2.trellis, AWGN(sigma), use_max=False)
    num_iter = 6
    decoder = HazzysTurboDecoder(decoder1, decoder2, interleaver, num_iter=num_iter)

    msg = turbo_encoder(input_bits)
    received_msg = channel(msg)
    tf_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vhazzys_turbo_decode(np_received, trellis1, trellis2, sigma ** 2, num_iter, commpy_interleaver)

    assert_array_almost_equal(commpy_L, tf_confidence.numpy(), decimal=5)

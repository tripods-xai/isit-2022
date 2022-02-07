import tensorflow as tf
import numpy as np
import commpy.channelcoding as cc

from src.channelcoding.channels import AWGN
from src.channelcoding.encoder_decoders import TurboNonsystematicEncoderDecoder, TurboSystematicEncoderDecoder
from src.channelcoding.encoders import AffineConvolutionalCode, TrellisCode
from src.channelcoding.bcjr import BCJRDecoder, HazzysTurboDecoder, SystematicTurboRepeater, TurboDecoder
from src.channelcoding.interleavers import PermuteInterleaver
from src.codes import turboae_binary_exact_nonsys
from tests.channelcoding.utils import FixedNPAWGN, FixedNoiseAWGN, NoNoiseAWGN, interleaver_to_commpy, vhazzys_turbo_decode, vsystematic_turbo_encode


# def test_channel_randomness():
#     sigma = 1
#     seed = 0

#     channel = FixedNoiseAWGN(sigma, seed)

#     compare_channel = FixedNPAWGN(sigma, seed)

#     msg_shape = (10, 20, 1)
#     np.testing.assert_array_almost_equal(channel(tf.zeros(msg_shape)).numpy(), channel(tf.zeros(msg_shape)).numpy())
#     np.testing.assert_array_almost_equal(channel(tf.zeros(msg_shape)).numpy(), compare_channel.corrupt(np.zeros(msg_shape)))

# def test_turbo_compare_with_commpy_one_iter_without_noise():
#     code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rc()
#     sigma = 1.
#     block_len = 100
#     num_iter = 1
#     seed = 0
#     tf.random.set_seed(seed)

#     np_msg = np.random.default_rng(seed+1).integers(0, 2, size=(100, 100, 1))
#     msg = tf.constant(np_msg, dtype=tf.float32)

#     # My Code
#     channel = NoNoiseAWGN(sigma)
#     # enc_dec = TurboSystematicEncoderDecoder(code1, code2, channel, block_len, False, num_iter)

#     systematic_code = code.with_systematic()
#     interleaved_code = code
#     use_max = False
#     interleaver = PermuteInterleaver(block_len)

#     # [Sys, Straight_1,..., Straight_{n-1}, Interleaved_1, ..., Interleaved_{m-1}]
#     encoder = systematic_code \
#         .concat(
#             interleaver.and_then(interleaved_code)
#         )
    
#     non_interleaved_bcjr = BCJRDecoder(
#         systematic_code.trellis,
#         channel, use_max=use_max
#     )
#     interleaved_bcjr = BCJRDecoder(
#         interleaved_code.trellis.with_systematic(),
#         channel, use_max=use_max
#     )

#     decoder = SystematicTurboRepeater(
#         num_noninterleaved_streams=non_interleaved_bcjr.num_input_channels, 
#         interleaver=interleaver
#     ).and_then(HazzysTurboDecoder(
#         decoder1=non_interleaved_bcjr,
#         decoder2=interleaved_bcjr,
#         interleaver=interleaver,
#         num_iter=num_iter
#     ))
#     encoder_decoder = encoder \
#         .and_then(channel) \
#         .and_then(decoder)

#     tf_encoded = encoder(msg)
#     tf_received = channel(tf_encoded)
#     tf_decode_L = decoder(tf_received)

#     # Commpy
#     commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
#     commpy_interleaver = interleaver_to_commpy(interleaver)
    
#     commpy_out = vsystematic_turbo_encode(np_msg, commpy_trellis, commpy_trellis, commpy_interleaver)
#     np_received = 2. * commpy_out - 1.
#     np_received_repeated = np.concatenate([np_received[:, :, 0:2], np_received[:, commpy_interleaver.p_array, 0:1], np_received[:, :, 2:3]], axis=-1)
#     commpy_L = vhazzys_turbo_decode(np_received_repeated, commpy_trellis, commpy_trellis, sigma ** 2, num_iter, commpy_interleaver)

#     np.testing.assert_array_almost_equal(tf_encoded.numpy(), commpy_out)
#     np.testing.assert_array_almost_equal(tf_received.numpy(), np_received)
#     np.testing.assert_array_almost_equal(tf_decode_L.numpy(), commpy_L, decimal=5)
#     np.testing.assert_array_almost_equal(encoder_decoder(msg).numpy(), commpy_L, decimal=5)

#     complete_encoder_decoder = TurboSystematicEncoderDecoder(
#         systematic_code,
#         interleaved_code,
#         channel,
#         HazzysTurboDecoder,
#         block_len,
#         use_max,
#         num_iter,
#         interleaver=interleaver
#     )
#     assert complete_encoder_decoder.rate == (1, 3)
#     np.testing.assert_array_almost_equal(complete_encoder_decoder(msg).numpy(), commpy_L, decimal=5)


# def test_turbo_compare_with_commpy_one_iter_with_noise():
#     code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rc()
#     sigma = 1.
#     block_len = 100
#     num_iter = 1
#     seed = 0
#     tf.random.set_seed(seed)

#     np_msg = np.random.default_rng(seed+1).integers(0, 2, size=(100, 100, 1))
#     msg = tf.constant(np_msg, dtype=tf.float32)

#     # My Code
#     channel = FixedNoiseAWGN(sigma, seed)
#     # enc_dec = TurboSystematicEncoderDecoder(code1, code2, channel, block_len, False, num_iter)

#     systematic_code = code.with_systematic()
#     interleaved_code = code
#     use_max = False
#     interleaver = PermuteInterleaver(block_len)

#     # [Sys, Straight_1,..., Straight_{n-1}, Interleaved_1, ..., Interleaved_{m-1}]
#     encoder = systematic_code \
#         .concat(
#             interleaver.and_then(interleaved_code)
#         )
    
#     non_interleaved_bcjr = BCJRDecoder(
#         systematic_code.trellis,
#         channel, use_max=use_max
#     )
#     interleaved_bcjr = BCJRDecoder(
#         interleaved_code.trellis.with_systematic(),
#         channel, use_max=use_max
#     )

#     decoder = SystematicTurboRepeater(
#         num_noninterleaved_streams=non_interleaved_bcjr.num_input_channels, 
#         interleaver=interleaver
#     ).and_then(HazzysTurboDecoder(
#         decoder1=non_interleaved_bcjr,
#         decoder2=interleaved_bcjr,
#         interleaver=interleaver,
#         num_iter=num_iter
#     ))
#     encoder_decoder = encoder \
#         .and_then(channel) \
#         .and_then(decoder)

#     tf_encoded = encoder(msg)
#     tf_received = channel(tf_encoded)
#     tf_decode_L = decoder(tf_received)

#     # Commpy
#     np_channel = FixedNPAWGN(sigma, seed)
#     commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
#     commpy_interleaver = interleaver_to_commpy(interleaver)
    
#     commpy_out = vsystematic_turbo_encode(np_msg, commpy_trellis, commpy_trellis, commpy_interleaver)
#     np_received = np_channel.corrupt(commpy_out)
#     np_received_repeated = np.concatenate([np_received[:, :, 0:2], np_received[:, commpy_interleaver.p_array, 0:1], np_received[:, :, 2:3]], axis=-1)
#     commpy_L = vhazzys_turbo_decode(np_received_repeated, commpy_trellis, commpy_trellis, sigma ** 2, num_iter, commpy_interleaver)

#     np.testing.assert_array_almost_equal(tf_encoded.numpy(), commpy_out)
#     np.testing.assert_array_almost_equal(tf_received.numpy(), np_received)
#     np.testing.assert_array_almost_equal(tf_decode_L.numpy(), commpy_L, decimal=5)
#     np.testing.assert_array_almost_equal(encoder_decoder(msg).numpy(), commpy_L, decimal=5)

#     complete_encoder_decoder = TurboSystematicEncoderDecoder(
#         systematic_code,
#         interleaved_code,
#         channel,
#         HazzysTurboDecoder,
#         block_len,
#         use_max,
#         num_iter,
#         interleaver=interleaver
#     )
#     assert complete_encoder_decoder.rate == (1, 3)
#     np.testing.assert_array_almost_equal(complete_encoder_decoder(msg).numpy(), commpy_L, decimal=5)

# def test_turbo_compare_with_commpy_two_iter_without_noise():
#     code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rc()
#     sigma = 1.
#     block_len = 100
#     num_iter = 2
#     seed = 0
#     tf.random.set_seed(seed)

#     np_msg = np.random.default_rng(seed+1).integers(0, 2, size=(100, 100, 1))
#     msg = tf.constant(np_msg, dtype=tf.float32)

#     # My Code
#     channel = NoNoiseAWGN(sigma)
#     # enc_dec = TurboSystematicEncoderDecoder(code1, code2, channel, block_len, False, num_iter)

#     systematic_code = code.with_systematic()
#     interleaved_code = code
#     use_max = False
#     interleaver = PermuteInterleaver(block_len)

#     # [Sys, Straight_1,..., Straight_{n-1}, Interleaved_1, ..., Interleaved_{m-1}]
#     encoder = systematic_code \
#         .concat(
#             interleaver.and_then(interleaved_code)
#         )
    
#     non_interleaved_bcjr = BCJRDecoder(
#         systematic_code.trellis,
#         channel, use_max=use_max
#     )
#     interleaved_bcjr = BCJRDecoder(
#         interleaved_code.trellis.with_systematic(),
#         channel, use_max=use_max
#     )

#     decoder = SystematicTurboRepeater(
#         num_noninterleaved_streams=non_interleaved_bcjr.num_input_channels, 
#         interleaver=interleaver
#     ).and_then(HazzysTurboDecoder(
#         decoder1=non_interleaved_bcjr,
#         decoder2=interleaved_bcjr,
#         interleaver=interleaver,
#         num_iter=num_iter
#     ))
#     encoder_decoder = encoder \
#         .and_then(channel) \
#         .and_then(decoder)

#     tf_encoded = encoder(msg)
#     tf_received = channel(tf_encoded)
#     tf_decode_L = decoder(tf_received)

#     # Commpy
#     commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
#     commpy_interleaver = interleaver_to_commpy(interleaver)
    
#     commpy_out = vsystematic_turbo_encode(np_msg, commpy_trellis, commpy_trellis, commpy_interleaver)
#     np_received = 2. * commpy_out - 1.
#     np_received_repeated = np.concatenate([np_received[:, :, 0:2], np_received[:, commpy_interleaver.p_array, 0:1], np_received[:, :, 2:3]], axis=-1)
#     commpy_L = vhazzys_turbo_decode(np_received_repeated, commpy_trellis, commpy_trellis, sigma ** 2, num_iter, commpy_interleaver)

#     np.testing.assert_array_almost_equal(tf_encoded.numpy(), commpy_out)
#     np.testing.assert_array_almost_equal(tf_received.numpy(), np_received)
#     np.testing.assert_array_almost_equal(tf_decode_L.numpy(), commpy_L, decimal=4)
#     np.testing.assert_array_almost_equal(encoder_decoder(msg).numpy(), commpy_L, decimal=4)

#     complete_encoder_decoder = TurboSystematicEncoderDecoder(
#         systematic_code,
#         interleaved_code,
#         channel,
#         HazzysTurboDecoder,
#         block_len,
#         use_max,
#         num_iter,
#         interleaver=interleaver
#     )
#     assert complete_encoder_decoder.rate == (1, 3)
#     np.testing.assert_array_almost_equal(complete_encoder_decoder(msg).numpy(), commpy_L, decimal=4)


# def test_turbo_compare_with_commpy_two_iter_with_noise():
#     code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rc()
#     sigma = 1.
#     block_len = 100
#     num_iter = 2
#     seed = 0
#     tf.random.set_seed(seed)

#     np_msg = np.random.default_rng(seed+1).integers(0, 2, size=(100, 100, 1))
#     msg = tf.constant(np_msg, dtype=tf.float32)

#     # My Code
#     channel = FixedNoiseAWGN(sigma, seed)
#     # enc_dec = TurboSystematicEncoderDecoder(code1, code2, channel, block_len, False, num_iter)

#     systematic_code = code.with_systematic()
#     interleaved_code = code
#     use_max = False
#     interleaver = PermuteInterleaver(block_len)

#     # [Sys, Straight_1,..., Straight_{n-1}, Interleaved_1, ..., Interleaved_{m-1}]
#     encoder = systematic_code \
#         .concat(
#             interleaver.and_then(interleaved_code)
#         )
    
#     non_interleaved_bcjr = BCJRDecoder(
#         systematic_code.trellis,
#         channel, use_max=use_max
#     )
#     interleaved_bcjr = BCJRDecoder(
#         interleaved_code.trellis.with_systematic(),
#         channel, use_max=use_max
#     )

#     decoder = SystematicTurboRepeater(
#         num_noninterleaved_streams=non_interleaved_bcjr.num_input_channels, 
#         interleaver=interleaver
#     ).and_then(HazzysTurboDecoder(
#         decoder1=non_interleaved_bcjr,
#         decoder2=interleaved_bcjr,
#         interleaver=interleaver,
#         num_iter=num_iter
#     ))
#     encoder_decoder = encoder \
#         .and_then(channel) \
#         .and_then(decoder)

#     tf_encoded = encoder(msg)
#     tf_received = channel(tf_encoded)
#     tf_decode_L = decoder(tf_received)

#     # Commpy
#     np_channel = FixedNPAWGN(sigma, seed)
#     commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
#     commpy_interleaver = interleaver_to_commpy(interleaver)
    
#     commpy_out = vsystematic_turbo_encode(np_msg, commpy_trellis, commpy_trellis, commpy_interleaver)
#     np_received = np_channel.corrupt(commpy_out)
#     np_received_repeated = np.concatenate([np_received[:, :, 0:2], np_received[:, commpy_interleaver.p_array, 0:1], np_received[:, :, 2:3]], axis=-1)
#     commpy_L = vhazzys_turbo_decode(np_received_repeated, commpy_trellis, commpy_trellis, sigma ** 2, num_iter, commpy_interleaver)

#     np.testing.assert_array_almost_equal(tf_encoded.numpy(), commpy_out)
#     np.testing.assert_array_almost_equal(tf_received.numpy(), np_received)
#     np.testing.assert_array_almost_equal(tf_decode_L.numpy(), commpy_L, decimal=4)
#     np.testing.assert_array_almost_equal(encoder_decoder(msg).numpy(), commpy_L, decimal=4)

#     complete_encoder_decoder = TurboSystematicEncoderDecoder(
#         systematic_code,
#         interleaved_code,
#         channel,
#         HazzysTurboDecoder,
#         block_len,
#         use_max,
#         num_iter,
#         interleaver=interleaver
#     )
#     assert complete_encoder_decoder.rate == (1, 3)
#     np.testing.assert_array_almost_equal(complete_encoder_decoder(msg).numpy(), commpy_L, decimal=4)

# def test_turbo_compare_with_commpy_six_iter_without_noise():
#     code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rc()
#     sigma = 1.
#     block_len = 100
#     num_iter = 6
#     seed = 0
#     tf.random.set_seed(seed)

#     np_msg = np.random.default_rng(seed+1).integers(0, 2, size=(100, 100, 1))
#     msg = tf.constant(np_msg, dtype=tf.float32)

#     # My Code
#     channel = NoNoiseAWGN(sigma)
#     # enc_dec = TurboSystematicEncoderDecoder(code1, code2, channel, block_len, False, num_iter)

#     systematic_code = code.with_systematic()
#     interleaved_code = code
#     use_max = False
#     interleaver = PermuteInterleaver(block_len)

#     # [Sys, Straight_1,..., Straight_{n-1}, Interleaved_1, ..., Interleaved_{m-1}]
#     encoder = systematic_code \
#         .concat(
#             interleaver.and_then(interleaved_code)
#         )
    
#     non_interleaved_bcjr = BCJRDecoder(
#         systematic_code.trellis,
#         channel, use_max=use_max
#     )
#     interleaved_bcjr = BCJRDecoder(
#         interleaved_code.trellis.with_systematic(),
#         channel, use_max=use_max
#     )

#     decoder = SystematicTurboRepeater(
#         num_noninterleaved_streams=non_interleaved_bcjr.num_input_channels, 
#         interleaver=interleaver
#     ).and_then(HazzysTurboDecoder(
#         decoder1=non_interleaved_bcjr,
#         decoder2=interleaved_bcjr,
#         interleaver=interleaver,
#         num_iter=num_iter
#     ))
#     encoder_decoder = encoder \
#         .and_then(channel) \
#         .and_then(decoder)

#     tf_encoded = encoder(msg)
#     tf_received = channel(tf_encoded)
#     tf_decode_L = decoder(tf_received)

#     # Commpy
#     commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
#     commpy_interleaver = interleaver_to_commpy(interleaver)
    
#     commpy_out = vsystematic_turbo_encode(np_msg, commpy_trellis, commpy_trellis, commpy_interleaver)
#     np_received = 2. * commpy_out - 1.
#     np_received_repeated = np.concatenate([np_received[:, :, 0:2], np_received[:, commpy_interleaver.p_array, 0:1], np_received[:, :, 2:3]], axis=-1)
#     commpy_L = vhazzys_turbo_decode(np_received_repeated, commpy_trellis, commpy_trellis, sigma ** 2, num_iter, commpy_interleaver)

#     np.testing.assert_array_almost_equal(tf_encoded.numpy(), commpy_out)
#     np.testing.assert_array_almost_equal(tf_received.numpy(), np_received)
#     np.testing.assert_array_almost_equal(tf_decode_L.numpy(), commpy_L, decimal=3)
#     np.testing.assert_array_almost_equal(encoder_decoder(msg).numpy(), commpy_L, decimal=3)

#     complete_encoder_decoder = TurboSystematicEncoderDecoder(
#         systematic_code,
#         interleaved_code,
#         channel,
#         HazzysTurboDecoder,
#         block_len,
#         use_max,
#         num_iter,
#         interleaver=interleaver
#     )
#     assert complete_encoder_decoder.rate == (1, 3)
#     np.testing.assert_array_almost_equal(complete_encoder_decoder(msg).numpy(), commpy_L, decimal=3)


# def test_turbo_compare_with_commpy_six_iter_with_noise():
#     code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rc()
#     sigma = 1.
#     block_len = 100
#     num_iter = 6
#     seed = 3
#     tf.random.set_seed(seed)

#     np_msg = np.random.default_rng(seed+1).integers(0, 2, size=(100, 100, 1))
#     msg = tf.constant(np_msg, dtype=tf.float32)

#     # My Code
#     channel = FixedNoiseAWGN(sigma, seed)
#     # enc_dec = TurboSystematicEncoderDecoder(code1, code2, channel, block_len, False, num_iter)

#     systematic_code = code.with_systematic()
#     interleaved_code = code
#     use_max = False
#     interleaver = PermuteInterleaver(block_len)

#     # [Sys, Straight_1,..., Straight_{n-1}, Interleaved_1, ..., Interleaved_{m-1}]
#     encoder = systematic_code \
#         .concat(
#             interleaver.and_then(interleaved_code)
#         )
    
#     non_interleaved_bcjr = BCJRDecoder(
#         systematic_code.trellis,
#         channel, use_max=use_max
#     )
#     interleaved_bcjr = BCJRDecoder(
#         interleaved_code.trellis.with_systematic(),
#         channel, use_max=use_max
#     )

#     decoder = SystematicTurboRepeater(
#         num_noninterleaved_streams=non_interleaved_bcjr.num_input_channels, 
#         interleaver=interleaver
#     ).and_then(HazzysTurboDecoder(
#         decoder1=non_interleaved_bcjr,
#         decoder2=interleaved_bcjr,
#         interleaver=interleaver,
#         num_iter=num_iter
#     ))
#     encoder_decoder = encoder \
#         .and_then(channel) \
#         .and_then(decoder)

#     tf_encoded = encoder(msg)
#     tf_received = channel(tf_encoded)
#     tf_decode_L = decoder(tf_received)

#     # Commpy
#     np_channel = FixedNPAWGN(sigma, seed)
#     commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
#     commpy_interleaver = interleaver_to_commpy(interleaver)
    
#     commpy_out = vsystematic_turbo_encode(np_msg, commpy_trellis, commpy_trellis, commpy_interleaver)
#     np_received = np_channel.corrupt(commpy_out)
#     np_received_repeated = np.concatenate([np_received[:, :, 0:2], np_received[:, commpy_interleaver.p_array, 0:1], np_received[:, :, 2:3]], axis=-1)
#     commpy_L = vhazzys_turbo_decode(np_received_repeated, commpy_trellis, commpy_trellis, sigma ** 2, num_iter, commpy_interleaver)

#     np.testing.assert_array_almost_equal(tf_encoded.numpy(), commpy_out)
#     np.testing.assert_array_almost_equal(tf_received.numpy(), np_received)
#     np.testing.assert_array_almost_equal(tf_decode_L.numpy(), commpy_L, decimal=3)
#     np.testing.assert_array_almost_equal(encoder_decoder(msg).numpy(), commpy_L, decimal=3)

#     complete_encoder_decoder = TurboSystematicEncoderDecoder(
#         systematic_code,
#         interleaved_code,
#         channel,
#         HazzysTurboDecoder,
#         block_len,
#         use_max,
#         num_iter,
#         interleaver=interleaver
#     )

#     assert complete_encoder_decoder.rate == (1, 3)
#     np.testing.assert_array_almost_equal(complete_encoder_decoder(msg).numpy(), commpy_L, decimal=2)

# def test_turboae_exact_nonsys_trellis_compare_one_iter():
#     block_len = 100
#     num_iter = 1
#     encoder_spec = turboae_binary_exact_nonsys()
#     tf_interleaver = PermuteInterleaver(block_len)

#     channel = FixedNoiseAWGN(1., 0)

#     conv_enc_dec = TurboNonsystematicEncoderDecoder(
#         encoder_spec.noninterleaved_code,
#         encoder_spec.interleaved_code,
#         channel,
#         TurboDecoder,
#         block_len,
#         False,
#         num_iter,
#         interleaver=tf_interleaver
#     )

#     trellis_enc_dec = TurboNonsystematicEncoderDecoder(
#         TrellisCode(encoder_spec.noninterleaved_code.trellis),
#         TrellisCode(encoder_spec.interleaved_code.trellis),
#         channel,
#         TurboDecoder,
#         block_len,
#         False,
#         num_iter,
#         interleaver=tf_interleaver
#     )
    
#     msg = tf.random.uniform((100, block_len, 1), dtype=tf.int32, maxval=2)
#     assert tf.reduce_all(conv_enc_dec(msg) == trellis_enc_dec(msg))

# def test_turboae_exact_nonsys_trellis_compare_two_iter():
#     block_len = 100
#     num_iter = 2
#     encoder_spec = turboae_binary_exact_nonsys()
#     tf_interleaver = PermuteInterleaver(block_len)

#     channel = FixedNoiseAWGN(1., 0)

#     conv_enc_dec = TurboNonsystematicEncoderDecoder(
#         encoder_spec.noninterleaved_code,
#         encoder_spec.interleaved_code,
#         channel,
#         TurboDecoder,
#         block_len,
#         False,
#         num_iter,
#         interleaver=tf_interleaver
#     )

#     trellis_enc_dec = TurboNonsystematicEncoderDecoder(
#         TrellisCode(encoder_spec.noninterleaved_code.trellis),
#         TrellisCode(encoder_spec.interleaved_code.trellis),
#         channel,
#         TurboDecoder,
#         block_len,
#         False,
#         num_iter,
#         interleaver=tf_interleaver
#     )
    
#     msg = tf.random.uniform((100, block_len, 1), dtype=tf.int32, maxval=2)
#     assert tf.reduce_all(conv_enc_dec(msg) == trellis_enc_dec(msg))

@tf.function
def run_enc_dec(enc_dec, msg):
    return enc_dec(msg)

def test_turboae_exact_nonsys_trellis_compare_six_iter():
    block_len = 100
    num_iter = 1
    encoder_spec = turboae_binary_exact_nonsys()
    tf_interleaver = PermuteInterleaver(block_len)

    channel = FixedNoiseAWGN(1., 0, (1000, block_len, 3))

    conv_enc_dec = TurboNonsystematicEncoderDecoder(
        encoder_spec.noninterleaved_code,
        encoder_spec.interleaved_code,
        channel,
        TurboDecoder,
        block_len,
        False,
        num_iter,
        interleaver=tf_interleaver
    )

    trellis_enc_dec = TurboNonsystematicEncoderDecoder(
        TrellisCode(encoder_spec.noninterleaved_code.trellis),
        TrellisCode(encoder_spec.interleaved_code.trellis),
        channel,
        TurboDecoder,
        block_len,
        False,
        num_iter,
        interleaver=tf_interleaver
    )
    
    msg = tf.random.uniform((1000, block_len, 1), dtype=tf.int32, maxval=2)
    conv_conf = run_enc_dec(conv_enc_dec, msg)
    trellis_conf = run_enc_dec(trellis_enc_dec, msg)
    assert tf.reduce_all(conv_conf == trellis_conf)

    conv_decoded = tf.cast(conv_conf > 0, dtype=tf.int32)
    conv_num_bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(msg, conv_decoded), tf.float32), axis=1)[:, 0]
    trellis_decoded = tf.cast(trellis_conf > 0, dtype=tf.int32)
    trellis_num_bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(msg, trellis_decoded), tf.float32), axis=1)[:, 0]
    print(f"Conv BER: {np.sum(conv_num_bit_errors.numpy()) / (100 * block_len)}")
    print(f"Trellis BER: {np.sum(trellis_num_bit_errors.numpy()) / (100 * block_len)}")
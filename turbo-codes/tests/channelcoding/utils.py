import tensorflow as tf
import commpy.channelcoding as cc
from commpy.channelcoding.convcode import conv_encode
import numpy as np
from tensor_annotations import tensorflow as ttf

from commpy.channelcoding.interleavers import RandInterlv
from src.channelcoding.channels import AWGN
from src.channelcoding.interleavers import PermuteInterleaver
from src.channelcoding.types import Time, Batch, Channels, A0, A1, A2, A3


def interleaver_to_commpy(interleaver: PermuteInterleaver):
    commpy_interleaver = RandInterlv(len(interleaver), seed=0)
    commpy_interleaver.p_array = interleaver.permutation.numpy()
    return commpy_interleaver

def turbo_encode(msg_bits, trellis1, trellis2, interleaver):
    """ Turbo Encoder.

    Encode Bits using a parallel concatenated rate-1/3
    turbo code consisting of two rate-1/2 systematic
    convolutional component codes.

    Parameters
    ----------
    msg_bits : 1D ndarray containing {0, 1}
        Stream of bits to be turbo encoded.

    trellis1 : Trellis object
        Trellis representation of the
        first two codes in the parallel concatenation.

    trellis2 : Trellis object
        Trellis representation of the
        first and interleaved code in the parallel concatenation.

    interleaver : Interleaver object
        Interleaver used in the turbo code.

    Returns
    -------
    [stream1, stream2, stream3] : list of 1D ndarrays
        Encoded bit streams corresponding
        to two non-interleaved outputs and one interleaved output.
    """
    stream = conv_encode(msg_bits, trellis1, termination='cont')
    stream1 = stream[::2]
    stream2 = stream[1::2]

    interlv_msg_bits = interleaver.interlv(msg_bits)
    interlv_stream = conv_encode(interlv_msg_bits, trellis2, termination='cont')
    stream1_i = interlv_stream[::2]
    stream3 = interlv_stream[1::2]

    assert len(stream1) == len(stream2) == len(stream3) == len(msg_bits)

    return [stream1, stream2, stream1_i, stream3]

def vsystematic_turbo_encode(msg, trellis1, trellis2, interleaver):
    def _turbo_encode(single_msg):
        outputs = turbo_encode(single_msg, trellis1, trellis2, interleaver)
        # print([len(o) for o in outputs])
        return np.stack([outputs[0], outputs[1], outputs[3]], axis=1)
    
    return np.apply_along_axis(_turbo_encode, axis=1, arr=msg[:, :, 0])

def turbo_decode(stream1, stream2, stream1_i, stream3, 
                 trellis1, trellis2,
                 noise_variance, number_iterations, interleaver, L_int = None):
    """ Turbo Decoder.

    Decodes a stream of convolutionally encoded
    (rate 1/3) bits using the BCJR algorithm.

    Parameters
    ----------
    stream1 : 1D ndarray
        Received symbols corresponding to
        the first output bits in the codeword.

    stream2 : 1D ndarray
        Received symbols corresponding to
        the second output bits in the codeword.

    stream3 : 1D ndarray
        Received symbols corresponding to the
        third output bits (interleaved) in the codeword.

    trellis1 : Trellis object
        Trellis representation of the non-interleaved convolutional codes
        used in the Turbo code.
    
    trellis2 : Trellis object
        Trellis representation of the interleaved convolutional codes
        used in the Turbo code.

    noise_variance : float
        Variance (power) of the AWGN channel.

    number_iterations : int
        Number of the iterations of the
        BCJR algorithm used in turbo decoding.

    interleaver : Interleaver object.
        Interleaver used in the turbo code.

    L_int : 1D ndarray
        Array representing the initial intrinsic
        information for all received
        symbols.

        Typically all zeros,
        corresponding to equal prior
        probabilities of bits 0 and 1.

    Returns
    -------
    decoded_bits : 1D ndarray of ints containing {0, 1}
        Decoded bit stream.

    """
    if L_int is None:
        L_int = np.zeros(len(stream1))

    L_int_1 = L_int

    # # Interleave stream 1 for input to second decoder
    # stream1_i = interleaver.interlv(stream1)

    iteration_count = 0
    max_iters = number_iterations
    while iteration_count < max_iters:
        [L_1, decoded_bits] = cc.map_decode(stream1, stream2,
                                             trellis1, noise_variance, L_int_1, 'compute')

        L_ext_1 = L_1 - L_int_1
        L_int_2 = interleaver.interlv(L_ext_1)

        # MAP Decoder - 2
        [L_2, decoded_bits] = cc.map_decode(stream1_i, stream3,
                                         trellis2, noise_variance, L_int_2, 'compute')
        L_ext_2 = L_2 - L_int_2
        L_int_1 = interleaver.deinterlv(L_ext_2)

        iteration_count += 1

    L_2_deinterleaved = interleaver.deinterlv(L_2)
    decoded_bits = (L_2_deinterleaved > 0).astype(int)

    return L_2_deinterleaved, decoded_bits


def vturbo_decode(msg, trellis1, trellis2, noise_variance, number_iterations, interleaver):
    def _turbo_decode(single_msg):
        str1, str2, str1_i, str3 = [single_msg[:, i] for i in range(4)]
        L_2_deinterleaved, decoded_bits = turbo_decode(str1, str2, str1_i, str3, trellis1, trellis2, noise_variance, number_iterations, interleaver)
        return L_2_deinterleaved[:, None]
    
    return np.stack([_turbo_decode(msg[i]) for i in range(msg.shape[0])], axis=0)

def hazzys_turbo_decode(str1, str2, str1_i, str3, trellis1, trellis2,
                 noise_variance, number_iterations, interleaver, L_int = None, fading=False):
    """ Turbo Decoder.

    Decodes a stream of convolutionally encoded
    (rate 1/3) bits using the BCJR algorithm.

    Parameters
    ----------
    sys_symbols : 1D ndarray
        Received symbols corresponding to
        the systematic (first output) bits in the codeword.

    non_sys_symbols_1 : 1D ndarray
        Received symbols corresponding to
        the first parity bits in the codeword.

    non_sys_symbols_2 : 1D ndarray
        Received symbols corresponding to the
        second parity bits in the codeword.

    trellis : Trellis object
        Trellis representation of the convolutional codes
        used in the Turbo code.

    noise_variance : float
        Variance (power) of the AWGN channel.

    number_iterations : int
        Number of the iterations of the
        BCJR algorithm used in turbo decoding.

    interleaver : Interleaver object.
        Interleaver used in the turbo code.

    L_int : 1D ndarray
        Array representing the initial intrinsic
        information for all received
        symbols.

        Typically all zeros,
        corresponding to equal prior
        probabilities of bits 0 and 1.

    Returns
    -------
    decoded_bits : 1D ndarray of ints containing {0, 1}
        Decoded bit stream.

    """
    if L_int is None:
        L_int = np.zeros(len(str1))

    L_int_1 = L_int

    # Interleave systematic symbols for input to second decoder
    # str1_i = interleaver.interlv(str1)

    weighted_sys = 2*str1*1.0/noise_variance # Is gonna be used in the final step of decoding. 

    iteration_count = 0
    max_iters = number_iterations
    while iteration_count < max_iters:
    # for iteration_count in range(number_iterations):
        prev_l1 = L_int_1

        # MAP Decoder - 1
        [L_ext_1, decoded_bits] = cc.map_decode(str1, str2,
                                             trellis1, noise_variance, L_int_1, 'compute')

        L_ext_1 = L_ext_1 - L_int_1
        L_ext_1 = L_ext_1 - weighted_sys
        
        L_int_2 = interleaver.interlv(L_ext_1)


        # MAP Decoder - 2
        [L_2, decoded_bits] = cc.map_decode(str1_i, str3,
                                         trellis2, noise_variance, L_int_2, 'compute')
        L_ext_2 = L_2 - L_int_2
        L_int_1 = interleaver.deinterlv(L_ext_2)
        L_int_1 = L_int_1 - weighted_sys
        
        iteration_count += 1

    L = L_ext_1 + L_int_1 + weighted_sys
    decoded_bits = L > 0

    return L, decoded_bits


def vhazzys_turbo_decode(msg, trellis1, trellis2, noise_variance, number_iterations, interleaver):
    def _turbo_decode(single_msg):
        str1, str2, str1_i, str3 = [single_msg[:, i] for i in range(4)]
        L_2_deinterleaved, decoded_bits = hazzys_turbo_decode(str1, str2, str1_i, str3, trellis1, trellis2, noise_variance, number_iterations, interleaver)
        return L_2_deinterleaved[:, None]
    
    return np.stack([_turbo_decode(msg[i]) for i in range(msg.shape[0])], axis=0)

def assert_binary_array(array):
    assert set(np.unique(array)).issubset({0, 1}), f"Array {array} was not binary"

class NoNoiseAWGN(AWGN):
    
    def call(self, msg: ttf.Tensor3[Batch, Time, Channels]) -> ttf.Tensor3[Batch, Time, Channels]:
        return 2. * msg - 1.

class FixedNoiseAWGN(AWGN):

    def __init__(self, sigma, seed, shape2):
        super().__init__(sigma)

        def noise_func(shape, sigma):
            return tf.constant(np.random.default_rng(seed).normal(0, sigma, size=shape2), dtype=tf.float32)

        self._noise_func = noise_func

class FixedNPAWGN:

    def __init__(self, sigma, seed):
        self.sigma = sigma
        self.seed = seed
    
    def corrupt(self, input_signal):
        
        assert_binary_array(input_signal)
        data_shape = input_signal.shape  # input_signal has to be a numpy array.

        noise = self.sigma * np.random.default_rng(self.seed).standard_normal(data_shape) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

        return corrupted_signal
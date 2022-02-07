# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

""" Turbo Codes """

from numpy import zeros
from commpy import channelcoding as cc

import math

def hazzys_turbo_decode(sys_symbols, non_sys_symbols_1, non_sys_symbols_2, trellis,
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
        L_int = zeros(len(sys_symbols))

    L_int_1 = L_int

    # Interleave systematic symbols for input to second decoder
    sys_symbols_i = interleaver.interlv(sys_symbols)

    weighted_sys = 2*sys_symbols*1.0/noise_variance # Is gonna be used in the final step of decoding. 

    for iteration_count in range(number_iterations):

        # MAP Decoder - 1
        [L_ext_1, decoded_bits] = cc.map_decode(sys_symbols, non_sys_symbols_1,
                                             trellis, noise_variance, L_int_1, 'compute')

        L_ext_1 = L_ext_1 - L_int_1
        L_ext_1 = L_ext_1 - weighted_sys
        
        L_int_2 = interleaver.interlv(L_ext_1)


        # MAP Decoder - 2
        [L_2, decoded_bits] = cc.map_decode(sys_symbols_i, non_sys_symbols_2,
                                         trellis, noise_variance, L_int_2, 'compute')
        L_ext_2 = L_2 - L_int_2
        L_int_1 = interleaver.deinterlv(L_ext_2)
        L_int_1 = L_int_1 - weighted_sys

    decoded_bits = (L_ext_1 + L_int_1 + weighted_sys > 0)

    return decoded_bits

if __name__ == "__main__":
    pass

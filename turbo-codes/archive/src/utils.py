__author__ = 'yihanjiang, abhmul'
'''
This is util functions of the decoder
'''

import numpy as np
import math
import os
import json
import dataclasses

def bitarray2dec(array, axis=0):
    return array.dot(1 << np.arange(array.shape[axis])[::-1]).astype(int)

dec2bitarray = np.vectorize(np.binary_repr)

def new_seed(rng):
    return rng.integers(99999999)

def binlen(num):
    return len(format(num, "b"))

def batched(array, original_dim):
     # Batch size 1
    return array[None] if array.ndim == original_dim else array

def hamming_dist(in_bitarray_1, in_bitarray_2, axis=None):
    """
    Computes the Hamming distance between two NumPy arrays of bits (0 and 1).

    Parameters
    ----------
    in_bit_array_1 : 1D ndarray of ints
        NumPy array of bits.

    in_bit_array_2 : 1D ndarray of ints
        NumPy array of bits.

    Returns
    -------
    distance : int
        Hamming distance between input bit arrays.
    """

    distance = np.bitwise_xor(in_bitarray_1, in_bitarray_2).sum(axis=axis)

    return distance

#######################################
# Noise Helper Function
#######################################
def awgn_corrupt(rng, input_signal, sigma = 1.0):
    '''
    Documentation TBD.
    :param sigma:
    :return:
    '''

    data_shape = input_signal.shape  # input_signal has to be a numpy array.

    noise = sigma * rng.standard_normal(data_shape) # Define noise
    corrupted_signal = 2.0*input_signal-1.0 + noise

    return corrupted_signal

#######################################
# Helper Function for convert SNR
#######################################

# SNR is -10 * log10(sigma**2)
# sigma is sqrt(10 ** (-snr / 10))
def snr2sigma(snr):
    return math.sqrt(10 ** (-snr / 10))

def sigma2snr(sigma):
    return -10 * math.log(sigma ** 2, 10)

def get_test_sigmas(snr_start, snr_end, snr_points):
    snrs = np.linspace(snr_start, snr_end, snr_points)
    test_sigmas = np.array([snr2sigma(snr) for snr in snrs])
    print('[testing] SNR range in dB ', snrs)
    print(f'[testing] Test sigmas are {test_sigmas}')
    print(f'[sanity check]: SNRs for sigmas are {[sigma2snr(sig) for sig in test_sigmas]}')
    
    return snrs, test_sigmas

# Helpers for checking inputs
def assert_binary_array(array):
    assert set(np.unique(array)) == {0, 1}, f"Array {array} was not binary"

# File utils
def safe_open_dir(dirpath):
    if not os.path.isdir(dirpath):
        print(f"Directory {dirpath} does not exist, creating it")
        os.makedirs(dirpath)
    return dirpath

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

if __name__ == '__main__':
    pass
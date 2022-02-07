__author__ = 'yihanjiang, abhmul'
'''
This is util functions of the decoder
'''

import numpy as np
import math
import os
import json
import dataclasses
import tensorflow as tf

def new_seed(rng):
    return rng.integers(99999999)

def batched(array, original_dim):
     # Batch size 1
    return array[None] if array.ndim == original_dim else array

def nullable(x):
    def null(*args, **kwargs): pass

    def inner(f):
        if x is not None:
            return f
        else:
            return null
    
    return inner

EPSILON = 1e-12

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

# File utils
def safe_open_dir(dirpath):
    if not os.path.isdir(dirpath):
        print(f"Directory {dirpath} does not exist, creating it")
        os.makedirs(dirpath)
    return dirpath

class TurboCodesJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if tf.is_tensor(o):
            return o.numpy().tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)

if __name__ == '__main__':
    pass
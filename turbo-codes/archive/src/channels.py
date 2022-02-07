import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from .utils import assert_binary_array

"""TF Channels"""
class TFChannel(layers.Layer):
    
    def call(self, input_signal):
        NotImplemented

class TFAWGN(TFChannel):
    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
    
    def call(self, input_signal):
        return input_signal + tf.random.normal(tf.shape(input_signal), stddev=self.sigma)


"""Np Channels"""
class Channel(object):

    def __init__(self):
        pass

    def corrupt(self, input_signal):
        NotImplemented

class AWGN(Channel):

    def __init__(self, sigma, rng):
        super().__init__()
        self.sigma = sigma
        self.rng = rng
    
    def corrupt(self, input_signal):
        
        assert_binary_array(input_signal)
        data_shape = input_signal.shape  # input_signal has to be a numpy array.

        noise = self.sigma * self.rng.standard_normal(data_shape) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

        return corrupted_signal
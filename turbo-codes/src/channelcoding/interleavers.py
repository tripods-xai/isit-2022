import abc

import tensorflow as tf

from src.channelcoding.dataclasses import FixedPermuteInterleaverSettings, RandomPermuteInterleaverSettings

from .codes import Code



class Interleaver(Code):
    @abc.abstractmethod
    def deinterleave(self, msg):
        pass
    
    def reset(self):
        pass
class FixedPermuteInterleaver(Interleaver):

    def __init__(self, block_len: int, permutation=None, depermutation=None, name: str = 'FixedPermuteInterleaver'):
        super().__init__(name)
        self.block_len = block_len
        if permutation is None:
            self.permutation = tf.random.shuffle(tf.range(block_len))
        else:
            self.permutation = permutation
        if depermutation is None:
            self.depermutation = tf.math.invert_permutation(self.permutation)
        else:
            # No validation is done
            self.depermutation = permutation
    
    @property
    def num_input_channels(self):
        return None
    
    @property
    def num_output_channels(self):
        return None
    
    def __len__(self):
        return self.block_len
    
    def call(self, msg):
        return tf.gather(msg, self.permutation, axis=1)
    
    def deinterleave(self, msg):
        return tf.gather(msg, self.depermutation, axis=1)
    
    def settings(self) -> FixedPermuteInterleaverSettings:
        return FixedPermuteInterleaverSettings(permutation=self.permutation, block_len=self.block_len, name=self.name)

class RandomPermuteInterleaver(Interleaver):

    def __init__(self, block_len: int, name: str = 'RandomPermuteInterleaver'):
        super().__init__(name)
        self.block_len = block_len
        self._permutation = None
        self._depermutation = None
    
    @property
    def num_input_channels(self):
        return None
    
    @property
    def num_output_channels(self):
        return None
    
    def __len__(self):
        return self.block_len
    
    def generate_permutations(self, batch_size):
        ta_perm = tf.TensorArray(tf.int32, size=batch_size, clear_after_read=True, element_shape=tf.TensorShape([self.block_len]))
        ta_deperm = tf.TensorArray(tf.int32, size=batch_size, clear_after_read=True, element_shape=tf.TensorShape([self.block_len]))
        for i in tf.range(batch_size):
            permutation = tf.random.shuffle(tf.range(self.block_len))
            ta_perm = ta_perm.write(i, permutation)
            ta_deperm = ta_deperm.write(i, tf.math.invert_permutation(permutation))
        return ta_perm.stack(), ta_deperm.stack()
    
    def set(self, msg):
        if self._permutation is None:
            batch_size = tf.shape(msg)[0]
            self._permutation, self._depermutation = self.generate_permutations(batch_size)
    
    def call(self, msg):
        self.set(msg)
        return tf.gather(msg, self._permutation, axis=1, batch_dims=1)
    
    def reset(self):
        self._permutation = None
        self._depermutation = None
    
    def deinterleave(self, msg):
        return tf.gather(msg, self._depermutation, axis=1, batch_dims=1)
    
    def settings(self) -> RandomPermuteInterleaverSettings:
        return RandomPermuteInterleaverSettings(block_len=self.block_len, name=self.name)


        
    
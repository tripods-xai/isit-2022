import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class Constraint(layers.Layer):
    def apply_last(self, x):
        NotImplemented
    
    def call(self, x, training=None):
        NotImplemented

def power_constrain(x, mean, std, center=0., scale=1.):
    return (x - mean) / (std + K.epsilon()) * scale + center 

class PowerConstraint(Constraint):

    def __init__(self, center=0., scale=1., **kwargs):
        super(PowerConstraint, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.reset_states()
        self.num_samples = tf.Variable(0.0)
        self.running_mean = tf.Variable(0.0)
        self.running_std = tf.Variable(0.0)

    def reset_states(self):
        self.num_samples = 0.
        self.running_mean = 0.
        self.running_std = 0.

    def call(self, x, training=None,):
        # Reduce over the batch
        mean = tf.math.reduce_mean(x)
        std = tf.math.reduce_std(x)
        
        if training or (training is None and K.learning_phase()):
            print(f"In training {K.learning_phase()}")
            return power_constrain(x, mean, std, self.center, self.scale), mean, std
        else:
            print(f"In testing {K.learning_phase()}")
            new_sample_count = tf.cast(tf.size(x), tf.float32)
            self.num_samples.assign_add(new_sample_count)
            self.running_mean = (self.running_mean * (self.num_samples - new_sample_count) + mean * new_sample_count) / self.num_samples
            self.running_std = tf.math.sqrt((self.running_std ** 2 * (self.num_samples - new_sample_count) + std ** 2 * new_sample_count) / self.num_samples)
            return power_constrain(x, self.running_mean, self.running_std, self.center, self.scale), self.running_mean, self.running_std
import abc
from os import name
from typing import overload
import tensorflow as tf
from src.utils import sigma2snr, snr2sigma
import tensorflow_probability as tfp
tfd = tfp.distributions

from src.channelcoding.dataclasses import AWGNSettings, AdditiveTonAWGNSettings, ChannelSettings, NonIIDMarkovianGaussianAsAWGNSettings, UnknownChannelSettings

from .codes import Code


def scale_constraint(data):
    return data * 2. - 1.

def identity_constraint(data):
    return data

class Channel(Code):

    @overload
    def log_likelihood(self, msg, outputs): ...
    @overload
    def log_likelihood(self, msg, outputs): ...
    @overload
    def log_likelihood(self, msg, outputs): ...
    @overload
    def log_likelihood(self, msg, outputs): ...
    @overload
    def log_likelihood(self, msg, outputs): ...

    @abc.abstractmethod
    def log_likelihood(self, msg, outputs):
        pass
    
    @abc.abstractmethod
    def logit_posterior(self, msg):
        pass

    @property
    def num_input_channels(self):
        return None
    
    @property
    def num_output_channels(self):
        return None
    
    def settings(self) -> ChannelSettings:
        return UnknownChannelSettings(self.name)

class NoisyChannel(Channel):

    @abc.abstractstaticmethod
    def noise_func(self, shape, *args):
        pass

class AWGN(NoisyChannel):
    def __init__(self, sigma, power_constraint=scale_constraint, name="AWGN"):
        super().__init__(name)
        self.sigma = sigma
        self.variance = self.sigma ** 2
        self.power_constraint = power_constraint
        # self._noise_constant = -tf.math.log(self.sigma * tf.math.sqrt(2 * math.pi))
    
    def noise_func(self, shape):
        return tf.random.normal(shape, stddev=self.sigma)
    
    def call(self, msg):
        # return (2. * msg - 1.) + self.noise_func(tf.shape(msg))
        return self.power_constraint(msg) + self.noise_func(tf.shape(msg))

    def log_likelihood(self, msg, outputs):
        msg_shape = tf.shape(msg)
        expanded_msg_shape = tf.concat([msg_shape[0:2], tf.ones((tf.rank(outputs) - 1,), dtype=tf.int32), msg_shape[2:3]], axis=0)
        msg = tf.reshape(msg, expanded_msg_shape)
        outputs = outputs[None, None]
        
        # Compute ln(Chi) values
        # chi_values[k, i, t] = log p(Y[k] | s[k] = i, s[k+1] = next_states[i, t])
        # B x K x 1 x 1 x ... x n - 1 x 1 x A0 x A1 x ... x n, reduce on last axis, result is B x K x A0 x A1 x ...
        # square_noise_sum = tf.math.reduce_sum(tf.square(msg - (2. * outputs - 1.)), axis=-1)
        square_noise_sum = tf.math.reduce_sum(tf.square(msg - self.power_constraint(outputs)), axis=-1)
        # chi_values = -tf.math.log(noise_std * tf.math.sqrt(2 * math.pi)) - 1 / (2 * noise_variance) * square_noise_sum
        # the first log term will cancel out in calculation of LLRs so I can drop it
        # return self._noise_constant - 1. / (2 * self.variance) * square_noise_sum
        return - 1. / (2 * self.variance) * square_noise_sum
    
    # @tf.function
    def logit_posterior(self, msg):
        return 2 * msg / self.variance
    
    def settings(self) -> ChannelSettings:
        return AWGNSettings.from_sigma(self.sigma, self.name)

class AdditiveTonAWGN(AWGN):
    def __init__(self, sigma, v=3, power_constraint=scale_constraint, name: str = "AdditiveTonAWGN"):
        super().__init__(sigma, power_constraint=power_constraint, name=name)
        self.v = v
        self.distribution = tfp.distributions.StudentT(df=self.v, loc=0, scale=1)
    
    def noise_func(self, shape):
        return self.sigma * tf.sqrt((self.v - 2) / self.v) * self.distribution.sample(shape)
    
    def settings(self) -> ChannelSettings:
        return AdditiveTonAWGNSettings.from_sigma(self.sigma, name)

# p_gg = 0.8         # stay in good state
#         p_bb = 0.8
#         bsc_k = snr_db2sigma(snr_sigma2db(this_sigma) + 1)          # accuracy on good state
#         bsc_h = snr_db2sigma(snr_sigma2db(this_sigma) - 1)   # accuracy on good state

#         fwd_noise = np.zeros(noise_shape)
#         for batch_idx in range(noise_shape[0]):
#             for code_idx in range(noise_shape[2]):

#                 good = True
#                 for time_idx in range(noise_shape[1]):
#                     if good:
#                         if test_sigma == 'default':
#                             fwd_noise[batch_idx,time_idx, code_idx] = bsc_k[batch_idx,time_idx, code_idx]
#                         else:
#                             fwd_noise[batch_idx,time_idx, code_idx] = bsc_k
#                         good = np.random.random()<p_gg
#                     elif not good:
#                         if test_sigma == 'default':
#                             fwd_noise[batch_idx,time_idx, code_idx] = bsc_h[batch_idx,time_idx, code_idx]
#                         else:
#                             fwd_noise[batch_idx,time_idx, code_idx] = bsc_h
#                         good = np.random.random()<p_bb
#                     else:
#                         print('bad!!! something happens')

#         fwd_noise = torch.from_numpy(fwd_noise).type(torch.FloatTensor)* torch.randn(noise_shape, dtype=torch.float)

class NonIIDMarkovianGaussianAsAWGN(AWGN):
    def __init__(self, sigma, block_len, p_gb=0.8, p_bg=0.8, power_constraint=scale_constraint, name="NonIIDMarkovianGaussianAsAWGN"):
        super().__init__(sigma, power_constraint=power_constraint, name=name)
        self.p_gb = p_gb
        self.p_bg = p_bg
        self.block_len = block_len

        self.snr = sigma2snr(self.sigma)
        self.good_snr = self.snr + 1
        self.bad_snr = self.snr - 1
        self.good_sigma = snr2sigma(self.good_snr)
        self.bad_sigma = snr2sigma(self.bad_snr)
        # Always start in good state; good = 1, bad = 0
        self.initial_distribution = tfd.Categorical(probs=[0.0, 1.0])
        self.transition_distribution = tfd.Categorical(probs=[[1 - p_bg, p_bg],
                                                              [p_gb, 1 - p_gb]])
        self.observation_distribution = tfd.Normal(loc=[0., 0.], scale=[self.bad_sigma, self.good_sigma])
        self.distribution = tfd.HiddenMarkovModel(self.initial_distribution, self.transition_distribution, self.observation_distribution, num_steps=self.block_len)
    
    def noise_func(self, shape):
        # shape[1] corresponds to time. we sample Batch x Channels x Time then swap channels and time axes
        return tf.transpose(self.distribution.sample((shape[0], shape[2])), perm=[0, 2, 1])
    
    def settings(self) -> NonIIDMarkovianGaussianAsAWGNSettings:
        return NonIIDMarkovianGaussianAsAWGNSettings(
            sigma=self.sigma,
            good_sigma=self.good_sigma,
            bad_sigma=self.bad_sigma,
            snr=self.snr,
            good_snr=self.good_snr,
            bad_snr=self.bad_snr,
            p_gb=self.p_gb,
            p_bg=self.p_bg,
            block_len=self.block_len,
            name=self.name
        )

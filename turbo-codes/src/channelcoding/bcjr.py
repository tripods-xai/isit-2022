import tensorflow as tf
import math
import numpy as np
import typing

import sys

from tensor_annotations import tensorflow as ttf

from src.channelcoding.encoders import TrellisCode

from .types import Time, Batch, Channels, PrevStates, States, Input
from .codes import Code
from .dataclasses import BCJRDecoderSettings, HazzysTurboDecoderSettings, StateTransitionGraph, Trellis, TurboDecoderSettings
from .interleavers import Interleaver
from .channels import Channel



# @tf.function
def backward_recursion(gamma_values, next_states, batch_size, K, S, reducer):
    # gamma_values = B x K x |S| x 2 : gamma_values[k, i, t] is the gamma for received k from state i to next_states[i, t]
    # next_states = |S| x 2 : next_states[i,t] is the next state after state i with input t
    # B[k][i] = log p(Y[k+1:K-1] | s[k+1] = i) 
    #         = log( Sum over t[ p(Y[k+2:K-1] | s[k+2] = next_states[i, t]) * p(Y[k+1], s[k+2] = next_states[i, t] | s[k+1] = i) ] )
    #         = logsumexp over t[ B[k+1, next_states[i, t]] + gamma_values[k+1, i, t] ]
    B = tf.TensorArray(tf.float32, size=K, clear_after_read=False)
    B = B.write(K-1, tf.zeros((batch_size, S)))
    for k in tf.range(K-2, -1, -1):
        # B x |S| x 2 + B x |S| x 2 -> B x |S|
        beta = reducer(gamma_values[:, k+1] + tf.gather(B.read(k+1), next_states, axis=1), 2)
        B = B.write(k, beta - reducer(beta, axis=1, keepdims=True))
    return tf.transpose(B.stack(), perm=[1, 0, 2])

# @tf.function
def forward_recursion(gamma_values: ttf.Tensor4[Batch, Time, States, Input], previous_states, batch_size, K, S, reducer):
    # previous_gamma_values = B x K x |S| x |Prev| : previous_gamma_values[:, k, i, t] is the gamma for received k from prev_states[i, t] to state i. |Prev| is ragged.
    # previous_states = |S| x |Prev| x 2 : previous_states[j] are the pairs of previous state that gets to j and the input to move to j. |Prev| is ragged.
    # A[k][j] = log p(Y[0:k-1], s[k] = j)
    #         = log( Sum over r[ p(Y[0:k-2], s[k-1]=previous_states[j, r, 0]) * p(Y[k-1], s[k]=j | s[k-1]=previous_states[j, r, 0]) ] )
    #         = logsumexp over r[ A[k-1, previous_states[j, r, 0]] + previous_gamma_values[k-1, j, r] ] ] 
    
    # States x |Prev| (ragged) x Batch x Time
    previous_gamma_values = tf.gather_nd(tf.transpose(gamma_values, perm=[2, 3, 0, 1]), previous_states)

    init_row = tf.tile(tf.constant([0.] + [-np.inf]*(S-1))[None, :], [batch_size, 1])
    A = tf.TensorArray(tf.float32, size=K, clear_after_read=False)
    A = A.write(0, init_row)
    for k in tf.range(1, K):
        # B x |S| x 2 + B x |S| x 2 -> B x |S|
        previous_alphas: ttf.Tensor2[Batch, States] = A.read(k-1)
        previous_gammas = previous_gamma_values[:, :, :, k-1]  # States x |Prev| x Batch
        # S x B

        alpha_not_trans = reducer(previous_gammas + tf.gather(tf.transpose(previous_alphas, perm=(1,0)), previous_states[:, :, 0], axis=0), axis=1)
        
        alpha_trans = tf.transpose(alpha_not_trans, perm=(1, 0))
        A = A.write(k, alpha_trans - reducer(alpha_trans, axis=1, keepdims=True))
    return tf.transpose(A.stack(), perm=[1, 0, 2])  # B x K x |S|

# Currently hardcoded for AWGN, but easy to change by passing in custom chi_values 
# @tf.function
def map_decode(
    received_symbols: ttf.Tensor3[Batch, Time, Channels], 
    next_states: ttf.Tensor2[States, Input], 
    previous_states: tf.RaggedTensor, # ttf.Tensor3[States, PrevStates, 2]
    L_int: ttf.Tensor2[Batch, Time], 
    chi_values: ttf.Tensor4[Batch, Time, States, Input], 
    use_max: bool=False
) -> ttf.Tensor2[Batch, Time]:
    if use_max:
        reducer = tf.math.reduce_max
    else:
        # reducer = tf.math.reduce_logsumexp
        # tf.math.reduce_logsumexp does not work with ragged tensors, use below instead
        # @tf.function
        def reducer(arr, axis, keepdims=False):
            raw_max = tf.reduce_max(arr, axis=axis, keepdims=True)
            my_max = tf.stop_gradient(
                tf.where(tf.math.is_finite(raw_max), raw_max, tf.zeros_like(raw_max))
            )
            result = tf.math.log(tf.math.reduce_sum(tf.math.exp(arr - my_max), axis=axis, keepdims=True)) + my_max
            # print(f"-2 Shape {result.shape}")
            # Reduce sum over a single item axis to apply keepdims conditionally without losing shape inference
            return tf.math.reduce_sum(result, axis=axis, keepdims=keepdims) 
            # return tf.cond(tf.constant(keepdims), true_fn=lambda: result, false_fn=lambda: tf.squeeze(result, axis=axis))

    
    # received_symbols = B x K x n : 1 / n is code rate
    # code_outputs = |S| x 2 x n : code_outputs[i,t] is the codeword emitted at state i with input t
    # next_states = |S| x 2 : next_states[i,t] is the next state after state i with input t
    # previous_states = |S| x ? x 2 : previous_states[j] are the pair of previous state that gets to j and the input to move to j
    # L_int = B x K : the prior LLR for x_k = 1
    batch_size = received_symbols.shape[0]
    K = received_symbols.shape[1]
    n = received_symbols.shape[2]
    S = next_states.shape[0]
    
    # Compute ln(Chi) values
    # chi_values[k, i, t] = log p(Y[k] | s[k] = i, s[k+1] = next_states[i, t])
    # B x K x 1 x 1 x n - 1 x 1 x |S| x 2 x n, reduce on 4th axis, result is B x K x |S| x 2
    # square_noise_sum = tf.math.reduce_sum(tf.square(received_symbols[:, :, None, None, :] - code_outputs[None, None, :, :, :]), axis=4)
    # chi_values = -tf.math.log(noise_std * tf.math.sqrt(2 * math.pi)) - 1 / (2 * noise_variance) * square_noise_sum
    # the first log term will cancel out in calculation of LLRs so I can drop it
    # chi_values = - 1. / (2 * noise_variance) * square_noise_sum
        
    # Compute ln(Gamma) values
    # gamma_values[k, i, t] = log p(Y[k], s[k+1] = next_states[i, t] | s[k] = i) = log p(s[k+1] = next_states[i, t] | s[k] = i) + chi_values[k, i, t]
    # B x K x 2
    transition_prob_values = tf.stack([tf.math.log_sigmoid(-L_int), tf.math.log_sigmoid(L_int)], axis=2)
    # B x K x |S| x 2
    gamma_values = chi_values + transition_prob_values[:, :, None, :]
    
    # Compute ln(B)
    # B x K x |S|
    B = backward_recursion(gamma_values, next_states, batch_size, K, S, reducer)
    
    # B x K x |S|
    A = forward_recursion(gamma_values, previous_states, batch_size, K, S, reducer)

    # Compute L_ext
    # L = log Sum over i[ p(Y[0:K-1], s_k=i, s_k+1=next_states[i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], Y[k], Y[k+1:K-1], s_k=i, s_k+1=next_states[i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k], s_k+1=next_states[i, 1] | s_k=i) * P(Y[k+1:K-1] | s_k+1=next_states[i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[i, 1], s_k=i) * p(s_k+1=next_states[i, 1] | s_k=i) * p(Y[k+1:K-1] | s_k+1=next_states[i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[i, 1], s_k=i) * p(x_k=1) * p(Y[k+1:K-1] | s_k+1=next_states[i, 1]) ] / "
    # = log( p(x_k=1) / p(x_k=0) ) * log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[i, 1], s_k=i) * p(Y[k+1:K-1] | s_k+1=next_states[i, 1]) ] / "
    # = L_int + logsumexp over i[ A[k, i] + chi_values[k, i, 1] + B[k, next_states[i, 1]] ] - logsumexp over i[ A[k, i] + chi_values[k, i, 0] + B[k, next_states[i, 1]] ] 
    # = L_int + L_ext
    # -> L_ext = logsumexp over i[ A[k, i] + chi_values[k, i, 1] + B[k, next_states[i, 1]] ] - logsumexp over i[ A[k, i] + chi_values[k, i, 0] + B[k, next_states[i, 1]] ]
    
    B_next_states = tf.gather(B, next_states, axis=2)
    # print(f"-1 Shape: {B_next_states}")
    L_ext = reducer(A + chi_values[:, :, :, 1] + B_next_states[:, :, :, 1], axis=2) - reducer(A + chi_values[:, :, :, 0] + B_next_states[:, :, :, 0], axis=2)
    # print(f"3 Shape {L_ext.shape}")
    return L_ext

class BCJRDecoder(Code):

    def __init__(self, trellis_code: TrellisCode, channel: Channel, use_max: bool = False, name: str='BCJRDecoder'):
        super().__init__(name)
        # self.trellis = trellis
        self.trellis_code = trellis_code
        self.channel = channel
        self.use_max = use_max

    @property
    def num_input_channels(self):
        return self.trellis_code.num_output_channels

    @property
    def num_output_channels(self):
        return 1
    
    # @tf.function
    def call(self, msg: ttf.Tensor3[Batch, Time, Channels]) -> ttf.Tensor3[Batch, Time, Channels]:
        # Channel 1 has priors, remaining channels are corrupted streams
        L_int = msg[:, :, 0]
        received_symbols = msg[:, :, 1:]
        # L_ext = map_decode(
        #     received_symbols=received_symbols, 
        #     next_states=self.trellis.state_transitions.next_states, 
        #     previous_states=self.trellis.state_transitions.previous_states, 
        #     L_int=L_int, 
        #     chi_values=self.channel.log_likelihood(received_symbols, self.trellis.output_table), 
        #     use_max=self.use_max
        # )
        L_ext = map_decode(
            received_symbols=received_symbols, 
            next_states=self.trellis_code.state_transiition.next_states,
            previous_states=self.trellis_code.state_transiition.previous_states, 
            L_int=L_int, 
            chi_values=self.channel.log_likelihood(received_symbols, self.trellis_code.output_table), 
            use_max=self.use_max
        )
        # print(f"0 Shape: {L_ext.shape}")
        return (L_int + L_ext)[:, :, None]
  
    # TODO: Remove?
    # def with_systematic_in(self) -> 'BCJRDecoder':
    #     return BCJRDecoder(
    #         trellis=self.trellis.with_systematic(),
    #         channel=self.channel,
    #         use_max=self.use_max
    #     )
    
    def settings(self):
        return BCJRDecoderSettings(
            trellis_code=self.trellis_code.settings(),
            channel=self.channel.settings(),
            use_max=self.use_max,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.num_output_channels,
            name=self.name
        )
    
    def training(self):
        self.use_max = True
    
    def validating(self):
        self.use_max = False

class PriorInjector(Code):

    def __init__(self, prior: ttf.Tensor2[Batch, Time]=None, name: str='PriorInjector'):
        super().__init__(name)
        self.prior = prior
    
    @property
    def num_input_channels(self):
        return None

    @property
    def num_output_channels(self):
        return None
    
    # A simple method, but enforces our requirement that the prior is in the first channel
    @staticmethod
    def inject_prior(prior: ttf.Tensor2[Batch, Time], msg: ttf.Tensor3[Batch, Time, Channels]) -> ttf.Tensor3[Batch, Time, Channels]:
        return tf.concat([prior[:, :, None], msg], axis=2)

    def call(self, msg: ttf.Tensor3[Batch, Time, Channels]) -> ttf.Tensor3[Batch, Time, Channels]:
        prior = tf.zeros((msg.shape[0], msg.shape[1])) if self.prior is None else self.prior
        return self.inject_prior(prior, msg)

class TurboDecoder(Code):
    
    def __init__(self, decoder1: BCJRDecoder, decoder2: BCJRDecoder, interleaver: Interleaver, num_iter: int=10, name="TurboDecoder"):
        super().__init__(name)
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.interleaver = interleaver
        self.num_iter = num_iter

        # decoder 1 tells us how much of the data is not interleaved
        self.num_noninterleaved_streams = self.decoder1.num_input_channels

        self.validate()
    
    @property
    def num_input_channels(self):
        return self.decoder1.num_input_channels + self.decoder2.num_input_channels

    @property
    def num_output_channels(self):
        return self.decoder1.num_output_channels
    
    def validate(self):
        if self.decoder1.num_output_channels is not None and self.decoder2.num_output_channels is not None:
            assert self.decoder1.num_output_channels == self.decoder2.num_output_channels
    
    # @tf.function
    def call(self, msg: ttf.Tensor3[Batch, Time, Channels]) -> ttf.Tensor3[Batch, Time, Channels]:
        # Msg comes in with channels [Straight_1, ..., Straight_n, Interleaved_1,..., Interleaved_m]
        # n = number of inputs to straight decoder
        # m = number of inputs to interleaved decoder
        batch_size = msg.shape[0]
        msg_len = msg.shape[1]
        msg_noninterleaved = msg[:, :, :self.num_noninterleaved_streams]
        msg_interleaved = msg[:, :, self.num_noninterleaved_streams:]

        L_int = tf.zeros((batch_size, msg_len, 1))

        return self.decode(msg_noninterleaved, msg_interleaved, L_int)

    # @tf.function
    def decode(self, msg_noninterleaved, msg_interleaved, L_int) -> ttf.Tensor3[Batch, Time, Channels]:
        L_int1 = L_int
        L_ext1 = tf.zeros_like(L_int1)
        for i in tf.range(self.num_iter):
            L_ext1 = self.decoder1(PriorInjector.inject_prior(L_int1[:, :, 0], msg_noninterleaved)) - L_int1

            L_int2 = self.interleaver(L_ext1)
            L_ext2 = self.decoder2(PriorInjector.inject_prior(L_int2[:, :, 0], msg_interleaved)) - L_int2

            L_int1 = self.interleaver.deinterleave(L_ext2)
        
        L = L_ext1 + L_int1
        return L
    
    def settings(self):
        return TurboDecoderSettings(
            decoder1=self.decoder1.settings(),
            decoder2=self.decoder2.settings(),
            interleaver=self.interleaver.settings(),
            num_iter=self.num_iter,
            num_noninterleaved_streams=self.num_noninterleaved_streams,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.num_output_channels,
            name=self.name
        )
    
    def training(self):
        self.decoder1.training()
        self.decoder2.training()
    
    def validating(self):
        self.decoder1.validating()
        self.decoder2.validating()

class SystematicTurboRepeater(Code):
    """Includes an interleaved systematic stream in the interleaved channels"""

    def __init__(self, num_noninterleaved_streams: int, interleaver: Interleaver, name: str = 'SystematicTurboRepeater'):
        super().__init__(name)
        self.num_noninterleaved_streams = num_noninterleaved_streams
        self.interleaver = interleaver
    
    @property
    def num_input_channels(self):
        return None

    @property
    def num_output_channels(self):
        return None

    # @tf.function
    def call(self, msg: ttf.Tensor3[Batch, Time, Channels]) -> ttf.Tensor3[Batch, Time, Channels]:
        # Msg comes in with channels [Sys, Straight_2, ..., Straight_n, Interleaved_1,...]
        # n = number of inputs to straight decoder
        # Msg leaves [Sys, Straight_2, ..., Straight_n, Interleaved_sys, Interleaved_1,...]
        return tf.concat([
            msg[:, :, :self.num_noninterleaved_streams],
            self.interleaver(msg[:, :, 0:1]),  # sys stream
            msg[:, :, self.num_noninterleaved_streams:]
        ], axis=2)
        

class HazzysTurboDecoder(TurboDecoder):

    def __init__(self, decoder1: BCJRDecoder, decoder2: BCJRDecoder, interleaver: Interleaver, num_iter: int=10, name: str = "HazzysTurboDecoder"):
        super().__init__(decoder1, decoder2, interleaver, num_iter, name)
        self.validate()
    
    def validate(self):
        assert type(self.decoder1.channel) is type(self.decoder2.channel)
    
    # @tf.function
    def call(self, msg: ttf.Tensor3[Batch, Time, Channels]) -> ttf.Tensor3[Batch, Time, Channels]:
        # Msg comes in with channels [Straight_1, ..., Straight_n, Interleaved_1,..., Interleaved_m]
        # n = number of inputs to straight decoder
        # m = number of inputs to interleaved decoder
        batch_size = msg.shape[0]
        msg_len = msg.shape[1]
        msg_noninterleaved = msg[:, :, :self.num_noninterleaved_streams]
        msg_interleaved = msg[:, :, self.num_noninterleaved_streams:]

        L_int = tf.zeros((batch_size, msg_len, 1))

        return self.decode(msg_noninterleaved, msg_interleaved, L_int)

    # @tf.function
    def decode(self, msg_noninterleaved, msg_interleaved, L_int) -> ttf.Tensor3[Batch, Time, Channels]:
        L_int1 = L_int
        L_ext1 = tf.zeros_like(L_int1)
        weighted_sys = self.decoder1.channel.logit_posterior(msg_noninterleaved[:, :, 0:1])
        for i in tf.range(self.num_iter):
            # tf.autograph.experimental.set_loop_options(
            #     shape_invariants=[
            #         (L_int1, L_int1.shape), 
            #         (L_ext1, tf.TensorShape([None, None, None])), 
            #         (weighted_sys, weighted_sys.shape)]
            #     )
            L_ext1 = self.decoder1(PriorInjector.inject_prior(L_int1[:, :, 0], msg_noninterleaved)) - L_int1 - weighted_sys

            L_int2 = self.interleaver(L_ext1)
            L_ext2 = self.decoder2(PriorInjector.inject_prior(L_int2[:, :, 0], msg_interleaved)) - L_int2

            L_int1 = self.interleaver.deinterleave(L_ext2) - weighted_sys
        
        L = L_ext1 + L_int1 + weighted_sys
        # print(self.decoder1.use_max)
        # print(self.decoder2.use_max)
        # L = weighted_sys + tf.reduce_mean(msg_noninterleaved) + tf.reduce_mean(msg_interleaved)
        # L = tf.reduce_mean(msg_interleaved, axis=2, keepdims=True) + tf.reduce_mean(msg_noninterleaved, axis=2, keepdims=True)
        return L

    def settings(self):
        return HazzysTurboDecoderSettings(
            decoder1=self.decoder1.settings(),
            decoder2=self.decoder2.settings(),
            interleaver=self.interleaver.settings(),
            num_iter=self.num_iter,
            num_noninterleaved_streams=self.num_noninterleaved_streams,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.num_output_channels,
            name=self.name
        )
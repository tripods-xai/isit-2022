from typing import List
import math


import tensorflow as tf

from .codes import Code
from .utils import bitarray2dec, safe_int, base_2_accumulator, dec2bitarray, enumerate_binary_inputs
from .dataclasses import Trellis, StateTransitionGraph, TrellisEncoderSettings


from ..utils import nullable, EPSILON

"""
Binary numbers are represented as big endian unless otherwise stated
Convolutional code state transition drops leftmost bit (most significant if big endian)
"""
class TrellisCode(Code):

    def __init__(self, trellis: Trellis, name: str = 'TrellisCode', normalize_output_table=False):
        super().__init__(name)
        self.trellis = trellis
        self.normalize_output_table = normalize_output_table
    
    @property
    def state_transiition(self):
        return self.trellis.state_transitions
    
    @property
    def next_states(self):
        return self.trellis.next_states
    
    @property
    def num_inputs(self):
        return self.trellis.num_inputs

    @property
    def num_states(self):
        return self.trellis.num_states
    
    @property
    def output_table(self):
        output_table = self.trellis.output_table
        if self.normalize_output_table:
            # print('before normalization')
            # print(output_table)
            # print('normalizing')
            output_table = (output_table - tf.reduce_mean(output_table, axis=[0, 1], keepdims=True)) / (EPSILON + tf.math.reduce_std(output_table, axis=[0, 1], keepdims=True))
            # print('after normalization')
            # print(output_table)
            return output_table
        else:
            return output_table
    
    def call(self, msg):
        """
        Assumes message is set of binary streams. Each channel is an individual stream
        """
        msg_reduced = bitarray2dec(tf.cast(msg, tf.int32), axis=2)
        msg_len = msg_reduced.shape[1]
        # Will be Time x Batch x Channels
        output: tf.TensorArray = tf.TensorArray(size=msg_len, dtype=tf.float32)
        # Will be Time x Batch - tensors are not stacked, discarded after reading
        states: tf.TensorArray = tf.TensorArray(size=msg_len+1, dtype=tf.int32)
        states = states.write(0, tf.zeros(msg_reduced.shape[0], dtype=tf.int32))
        # with tf.GradientTape(persistent=True) as tape:
        for t in tf.range(msg_len):
            ind_tensor = tf.stack([states.read(t), msg_reduced[:, t]], axis=1) 
            output = output.write(t, tf.gather_nd(self.output_table, ind_tensor))
            states = states.write(t+1, tf.gather_nd(self.next_states, ind_tensor))
        states = states.close()
        output_result = tf.transpose(output.stack(), perm=[1, 0, 2])
        # print('Internal Watched:')
        # print(tape.watched_variables())
        # print('Internal gradient')
        # print(tape.gradient(output_result, self.output_table))
        return output_result
        # return tf.transpose(output.stack(), perm=[1, 0, 2])
    
    def concat(self, code2: Code) -> Code:
        if isinstance(code2, TrellisCode) and self.trellis._check_state_table_compatibility(code2.trellis):
            concat_trellis = self.trellis.concat(code2.trellis)
            # TODO figure this out
            return TrellisCode(concat_trellis, name='_'.join([self.name, code2.name]), normalize_output_table=(self.normalize_output_table and code2.normalize_output_table))
        else:
            return super().concat(code2)
    
    def with_systematic(self) -> 'TrellisCode':
        return TrellisCode(self.trellis.with_systematic(), name='_'.join([self.name, 'systematic']), normalize_output_table=self.normalize_output_table)
    
    def __mul__(self, other: object):
        return TrellisCode(self.trellis * other)
    
    def __add__(self, other: object):
        return TrellisCode(self.trellis + other)
    
    def __sub__(self, other: object):
        return TrellisCode(self.trellis - other)
    
    def __truediv__(self, other: object):
        return TrellisCode(self.trellis / other)
    
    def settings(self) -> TrellisEncoderSettings:
        return TrellisEncoderSettings(
            trellis=self.trellis,
            num_states=self.num_states,
            num_inputs=self.num_inputs,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.num_output_channels,
            normalize_output_table=self.normalize_output_table,
            name=self.name
        )
    
    def training(self):
        self.trellis = self.trellis.training()
    
    @property
    def num_input_channels(self):
        return math.ceil(math.log2(self.num_inputs))
    
    @property
    def num_output_channels(self):
        return self.trellis.num_outputs
    
    def parameters(self) -> List[tf.Variable]:
        return [self.trellis.output_table] # type: ignore

"""With these, the more efficient implementations may work, but I'll fall back to trellis code if there's bugs.
It will fall back anyways when I add the systematic code.
"""
class GeneralizedConvolutionalCode(TrellisCode):
    """
    Convolutional code represented by a table
    feedback table is required to be a binary table
    """
    def __init__(self, table, feedback = None, name: str = 'GeneralizedConvolutionalCode'):
        self.table = table
        self.feedback = feedback

        self.window = safe_int(math.log2(self.num_possible_windows))
        self._base_2 = base_2_accumulator(self.window)

        trellis = self._construct_trellis()

        super().__init__(trellis, name=name)

        self.validate()
    
    @property
    def num_possible_windows(self):
        return self.table.shape[0]
    
    def validate(self):
        assert self.num_input_channels == 1  # These convolutional codes are fixed to only take in 1 input bit
        if self.feedback is not None:
            assert self.num_possible_windows == self.feedback.shape[0]
        # Debugging type checks
        tf.debugging.assert_type(self.table, tf_type=tf.float32)
        nullable(self.feedback)(tf.debugging.assert_type)(self.feedback, tf_type=tf.int32)
        
    def _construct_trellis(self) -> Trellis:
        binary_states = dec2bitarray(
            tf.range(self.num_possible_windows), num_bits=self.window
        )
        if self.feedback is not None:
            binary_states = tf.concat(
                [binary_states[:, :-1], self.feedback[:, None]], axis=1
            )
        
        # Even indicies correspond to the last bit not included in state. 
        # Feedback will change this bit
        reordered_table = tf.gather(self.table, bitarray2dec(binary_states))
        output_table = tf.stack(
            [reordered_table[::2], reordered_table[1::2]], axis=1
        )

        next_states = bitarray2dec(binary_states[:, 1:], axis=-1)
        next_states_table = tf.stack(
            [next_states[::2], next_states[1::2]], axis=1
        )
        
        return Trellis(
            state_transitions=StateTransitionGraph.from_next_states(next_states_table), 
            output_table=output_table
        )

    # def call(self, msg):
    #     """
    #     Assumes message is a binary stream (1 channel).
    #     """
    #     # return super().call(msg)
    #     # TODO: Don't use optimizations for now, TF bug: https://github.com/tensorflow/tensorflow/issues/53489
    #     # if self.feedback is not None:
    #     #     return super().__call__(msg)
    #     # else:
    #     #     # Call the parallelizeable encoding implementation if we have no feedback
    #     #     print("optimized")
    #     #     return self.conv_encode(msg)

    # TODO: Optimization, not used
    def _construct_state_sequence(self, msg):
        steps = msg.shape[1]
        msg_prepended = tf.pad(msg, paddings=tf.constant([[0, 0], [self.window-1, 0]]))
        # Convolve base-2 transformer over to efficiently get states
        conv_msg_input = tf.cast(msg_prepended[:,:,None], dtype=tf.float32)
        base_2_filter = tf.cast(self._base_2[:, None, None], dtype=tf.float32)
        return tf.cast(tf.nn.conv1d(conv_msg_input, base_2_filter, stride=1, padding='VALID'), dtype=tf.int32)[:,:,0]

    # TODO: Optimization, not used
    def conv_encode(self, msg):
        msg_channel_0 = tf.ensure_shape(msg, (None, None, self.num_input_channels))[:, :, 0]
        
        input_sequence = self._construct_state_sequence(msg_channel_0)
        return tf.gather(self.table, input_sequence, axis=0)

    # TODO: Optimization, not used
    def concat(self, code2: Code) -> Code:
        # If we know it's another convcode and they have the same feedback, we can optimize
        # if isinstance(code2, GeneralizedConvolutionalCode) and self._check_feedback_compatibility(code2):
        #     return self.concat_convcode(code2)
        # else:
        #     return super().concat(code2)
        # We'll just rely on trellis codes for now
        return super().concat(code2)
    
    # TODO: These are currently not used because we just fall back to TrellisCode. Assess their usefulness
    def _check_feedback_compatibility(self, code2: 'GeneralizedConvolutionalCode') -> bool:
        return (self.feedback is None and code2.feedback is None) or \
            tf.reduce_all(tf.equal(self.feedback, code2.feedback))
    
    # TODO: Optimization, not used
    def concat_convcode(self, code2: 'GeneralizedConvolutionalCode') -> 'GeneralizedConvolutionalCode':
        joined_table = tf.concat([self.table, code2.table], axis=1)
        return GeneralizedConvolutionalCode(joined_table, feedback=self.feedback)
    
    def _check_recursive_condition(self):
        if self.feedback is not None:
            raise ValueError(f"Cannot create recursive code, code already has feedback")
        if not tf.reduce_all(tf.logical_or(self.output_table[:, :, 0] == 0.0, self.output_table[:, :, 0] == 1.0)):
            raise ValueError(f"First channel is not binary out.")
        if not tf.reduce_all(self.output_table[:, 0, 0] != self.output_table[:, 1, 0]):
            raise ValueError(f"Cannot invert code, some output does not change when input changes: 0 -> {self.output_table[:, 0, 0]} 1 -> {self.output_table[:, 1, 0]}")

    def to_rc(self):
        self._check_recursive_condition()
        feedback = tf.cast(self.table[:, 0], dtype=tf.int32)
        return GeneralizedConvolutionalCode(table=self.table[:, 1:], feedback=feedback)
    
    def to_rsc(self):
        return self.to_rc().with_systematic()


class AffineConvolutionalCode(GeneralizedConvolutionalCode):
    """Convolutional code represented by a single boolean affine function"""
    def __init__(self, generator, bias, name: str = 'AffineConvolutionalCode'):
        self.validate_inputs(generator, bias)

        self.generator = generator
        self.bias = bias
        
        window = self.generator.shape[1]

        self._generator_filter = tf.cast(tf.transpose(self.generator, perm=[1, 0])[:, None, :], dtype=tf.float32)

        # Create the table
        self.code_inputs = enumerate_binary_inputs(window)
        table = self._encode(self.code_inputs)[:, 0, :]
        
        super().__init__(table, name=name)

    
    def validate_inputs(self, generator, bias):
        tf.debugging.assert_type(generator, tf_type=tf.int32)
        tf.debugging.assert_type(bias, tf_type=tf.int32)
        assert generator.shape[0] == bias.shape[0]
    
    # def call(self, msg):
    #     """
    #     Assumes message is a binary stream (1 channel).
    #     """
    #     return super().call(msg)
        # TODO: Don't use optimizations for now
        # msg_channel_0 = tf.ensure_shape(msg, (None, None, self.k))[:, :, 0]
        # msg_prepended = tf.pad(msg_channel_0, paddings=tf.constant([[0, 0], [self.window-1, 0]]))
        # return self._encode(msg_prepended)
        
    # TODO: Optimization, not used
    def _encode(self, msg):
        """
        Actual logic for encoding, is not responsible for padding with initial state.
        Assumes that the message only has 1 channel (and thus channel dimension is reduced out).
        """
        # Convolve generator over message to efficiently get outputs
        conv_msg_input = tf.cast(msg[:,:,None], dtype=tf.float32)
        conv_output = tf.cast(tf.nn.conv1d(conv_msg_input, self._generator_filter, stride=1, padding='VALID'), dtype=tf.int32)
        return tf.cast((conv_output + self.bias[None, None, :]) % 2, dtype=tf.float32)
    
    def to_rc(self):
        feedback = tf.cast(self.table[:, 0], tf.int32)
        table = self.table[:, 1:]
        return GeneralizedConvolutionalCode(table, feedback)

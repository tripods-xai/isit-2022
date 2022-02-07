import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from pprint import pprint

import itertools as it

from .constraints import power_constrain
from .channels import TFAWGN
from .utils import bitarray2dec, assert_binary_array


class ConvEncoder(layers.Layer):

    def __init__(self, n_outputs, window):
        super(ConvEncoder, self).__init__()
        self.n_outputs = n_outputs
        self.window = window
    
    def call(self, inputs, pad=True):
        """
        Encoder inputs come in as binary float B x N tf array
        Encoder is responsible for spitting n_output B x N streams out as a list. This means it must
        - pad input stream to start with 0 state
        """
        NotImplemented

class SystematicConvEncoder(ConvEncoder):

    def __init__(self, n_outputs=3, hidden_layers=1, hidden_dim=1000, kernel_dims=(3,1,)):
        assert len(kernel_dims) - 1 == hidden_layers
        window = kernel_dims[0] + sum(k - 1 for k in kernel_dims[1:])
        super(SystematicConvEncoder, self).__init__(n_outputs, window)
        self.n_outputs = n_outputs
        self.kernel_dims = kernel_dims
        self.window = window
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

        self.output_layer_outputs = n_outputs - 1
        self.layer_dims = [hidden_dim] * hidden_layers + [self.output_layer_outputs]
        self.activations = ['relu'] * hidden_layers + ['linear']
        self.layer_names = [f'processor_{i}' for i in range(hidden_layers)] + ['output_layer']
        self.conv_layers = [layers.Conv1D(h, k, activation=activation, name=name) for (h, k, activation, name) in zip(self.layer_dims, self.kernel_dims, self.activations, self.layer_names)]
        self.batch_norm_layers = [layers.BatchNormalization() for i in range(hidden_layers)]

    def call(self, inputs, pad=True):
        """
        Encoder inputs come in as binary float B x N tf array
        Encoder is responsible for spitting n_output B x N streams out as a list. This means it must
        - reshape the inputs to go through a convolutional layer (with 1 filter)
        - center input stream to [-1, 1]
        - pad input stream to start with 0 state for conv layer
        - remove last filter dimension before returning
        """
        # Rescale binary input to be 1 or -1
        # Inputs are B x N
        systematic = 2. * inputs - 1.
        # add filter dimension
        conv_input = systematic[:, :, None]
        # Sometimes we need to add the initial state, sometimes we don't
        conv_input = tf.pad(conv_input, [[0, 0], [self.window - 1, 0], [0, 0]], constant_values=-1.) if pad else conv_input
        systematic_out = systematic if pad else systematic[:, self.window-1:]

        x = conv_input
        for i in range(self.hidden_layers):
            x = self.conv_layers[i](x)
            x = self.batch_norm_layers[i](x)
        return [systematic_out,] + tf.unstack(self.conv_layers[-1](x), axis=2)

    def get_config(self):
        return {"n_outputs": self.n_outputs, 'window': self.window, 'hidden_layers': self.hidden_layers, 'hidden_dim': self.hidden_dim, 'kernel_dims': self.kernel_dims}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class NonsystematicConvEncoder(ConvEncoder):

    def __init__(self, n_outputs=3, hidden_layers=1, hidden_dim=1000, kernel_dims=(3,1,)):
        assert len(kernel_dims) - 1 == hidden_layers
        window = kernel_dims[0] + sum(k - 1 for k in kernel_dims[1:])
        super().__init__(n_outputs, window)
        self.n_outputs = n_outputs
        self.kernel_dims = kernel_dims
        self.window = window
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

        self.output_layer_outputs = n_outputs
        self.layer_dims = [hidden_dim] * hidden_layers + [self.output_layer_outputs]
        self.activations = ['relu'] * hidden_layers + ['linear']
        self.layer_names = [f'processor_{i}' for i in range(hidden_layers)] + ['output_layer']
        self.conv_layers = [layers.Conv1D(h, k, activation=activation, name=name) for (h, k, activation, name) in zip(self.layer_dims, self.kernel_dims, self.activations, self.layer_names)]
        self.batch_norm_layers = [layers.BatchNormalization() for i in range(hidden_layers)]

    def call(self, inputs, pad=True):
        """
        Encoder inputs come in as binary float B x N tf array
        Encoder is responsible for spitting n_output B x N streams out as a list. This means it must
        - reshape the inputs to go through a convolutional layer (with 1 filter)
        - center input stream to [-1, 1]
        - pad input stream to start with 0 state for conv layer
        - remove last filter dimension before returning
        """
        # Rescale binary input to be 1 or -1
        # Inputs are B x N
        systematic = 2. * inputs - 1.
        # add filter dimension
        conv_input = systematic[:, :, None]
        # Sometimes we need to add the initial state, sometimes we don't
        conv_input = tf.pad(conv_input, [[0, 0], [self.window - 1, 0], [0, 0]], constant_values=-1.) if pad else conv_input
        systematic_out = systematic if pad else systematic[:, self.window-1:]

        x = conv_input
        for i in range(self.hidden_layers):
            x = self.conv_layers[i](x)
            x = self.batch_norm_layers[i](x)
        return tf.unstack(self.conv_layers[-1](x), axis=2)

    def get_config(self):
        return {"n_outputs": self.n_outputs, 'window': self.window, 'hidden_layers': self.hidden_layers, 'hidden_dim': self.hidden_dim, 'kernel_dims': self.kernel_dims}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TurboNonsystematicSharedEncoder(layers.Layer):

    def __init__(self, shared_encoder, constraint, trellis_helper, tf_interleaver):
        super().__init__()
        assert shared_encoder.n_outputs == 3
        self.shared_encoder = shared_encoder
        self.constraint = constraint
        self.trellis_helper = trellis_helper
        self.tf_interleaver = tf_interleaver
    
    def call(self, input_stream, noisy_channel):
        # Input stream should have size B x block_len
        noninterleaved_input = input_stream
        interleaved_input = tf.gather(noninterleaved_input, self.tf_interleaver["permutation"], axis=1)
        combined_input = tf.concat([noninterleaved_input, interleaved_input], axis=0)
        
        par0, par1, par2 = self.shared_encoder(combined_input)
        non_i_par0 = par0[:noninterleaved_input.shape[0]]
        non_i_par1 = par1[:noninterleaved_input.shape[0]]
        i_par0 = par0[noninterleaved_input.shape[0]:]
        i_par2 = par2[noninterleaved_input.shape[0]:]

        # Apply the constraint
        constrained_outputs, mean, std = self.constraint(tf.stack([non_i_par0, non_i_par1, i_par0, i_par2], axis=0))
        non_i_par0 = constrained_outputs[0]
        non_i_par1 = constrained_outputs[1]
        i_par0 = constrained_outputs[2]
        i_par2 = constrained_outputs[3]
        
        # Get the output tables
        code1_output_table, code2_output_table = self.get_code_outputs(mean, std)

        # Corrupt the streams
        non_i_par0_corrupted = noisy_channel(non_i_par0)
        non_i_par1_corrupted = noisy_channel(non_i_par1)
        i_par2_corrupted = noisy_channel(i_par2)
        i_par0_corrupted = noisy_channel(i_par0)

        # DEBUG Do not corrupt the streams
        # non_i_sys_corrupted = non_i_sys
        # non_i_par_corrupted = non_i_par
        # i_par_corrupted = i_par
        # i_sys_corrupted = tf.gather(non_i_sys_corrupted, self.tf_interleaver["permutation"], axis=1)

        # Ship the completed package to the decoder
        msg = {
            "stream1": non_i_par0_corrupted, 
            "stream2": non_i_par1_corrupted, 
            "stream1_i": i_par0_corrupted, 
            "stream3": i_par2_corrupted,
            "interleaver": self.tf_interleaver,
            "trellis1": {
                "next_states": self.trellis_helper.tf_next_states,
                "previous_states": self.trellis_helper.tf_prev_states,
                "code_outputs": code1_output_table,
            },
            "trellis2": {
                "next_states": self.trellis_helper.tf_next_states,
                "previous_states": self.trellis_helper.tf_prev_states,
                "code_outputs": code2_output_table,
            },
        }

        return msg

    def get_code_outputs(self, mean, std):
        # Each code output is B x 1 arranged corresponding to possible_inputs
        par0_outputs, par1_outputs, par2_outputs = self.shared_encoder(self.trellis_helper.tf_possible_inputs, pad=False)
        par0_outputs = power_constrain(par0_outputs, mean, std)
        par1_outputs = power_constrain(par1_outputs, mean, std)
        par2_outputs = power_constrain(par2_outputs, mean, std)

        # Built code output table
        code1_output_table = tf.reshape(tf.concat([par0_outputs, par1_outputs], axis=1), (self.trellis_helper.num_states, self.trellis_helper.num_inputs, -1))
        code2_output_table = tf.reshape(tf.concat([par0_outputs, par2_outputs], axis=1), (self.trellis_helper.num_states, self.trellis_helper.num_inputs, -1))
        return code1_output_table, code2_output_table

class TurboSystematicSeparateEncoder(layers.Layer):

    def __init__(self, encoder1, encoder2, constraint1, constraint2, trellis_helper, tf_interleaver):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.constraint1 = constraint1
        self.constraint2 = constraint2
        self.trellis_helper = trellis_helper
        self.tf_interleaver = tf_interleaver
    
    def call(self, input_stream, noisy_channel):
        # Input stream should have size B x block_len
        noninterleaved_input = input_stream
        interleaved_input = tf.gather(noninterleaved_input, self.tf_interleaver["permutation"], axis=1)
        
        non_i_sys, non_i_par = self.encoder1(noninterleaved_input)
        _, i_par = self.encoder2(interleaved_input)

        # Apply the constraint
        non_i_par, mean1, std1 = self.constraint1(non_i_par)
        i_par, mean2, std2 = self.constraint2(i_par)
        
        # Get the output tables
        code1_output_table, code2_output_table = self.get_code_outputs(mean1, std1, mean2, std2)

        # Corrupt the streams
        non_i_sys_corrupted = noisy_channel(non_i_sys)
        non_i_par_corrupted = noisy_channel(non_i_par)
        i_par_corrupted = noisy_channel(i_par)
        i_sys_corrupted = tf.gather(non_i_sys_corrupted, self.tf_interleaver["permutation"], axis=1)

        # DEBUG Do not corrupt the streams
        # non_i_sys_corrupted = non_i_sys
        # non_i_par_corrupted = non_i_par
        # i_par_corrupted = i_par
        # i_sys_corrupted = tf.gather(non_i_sys_corrupted, self.tf_interleaver["permutation"], axis=1)

        # Ship the completed package to the decoder
        msg = {
            "stream1": non_i_sys_corrupted, 
            "stream2": non_i_par_corrupted, 
            "stream1_i": i_sys_corrupted, 
            "stream3": i_par_corrupted,
            "interleaver": self.tf_interleaver,
            "trellis1": {
                "next_states": self.trellis_helper.tf_next_states,
                "previous_states": self.trellis_helper.tf_prev_states,
                "code_outputs": code1_output_table,
            },
            "trellis2": {
                "next_states": self.trellis_helper.tf_next_states,
                "previous_states": self.trellis_helper.tf_prev_states,
                "code_outputs": code2_output_table,
            },
        }

        return msg

    def get_code_outputs(self, mean1, std1, mean2, std2):
        # Each code output is B x 1 arranged corresponding to possible_inputs
        sys_outputs, par1_outputs = self.encoder1(self.trellis_helper.tf_possible_inputs, pad=False)
        _, par2_outputs = self.encoder2(self.trellis_helper.tf_possible_inputs, pad=False)
        par1_outputs = power_constrain(par1_outputs, mean1, std1)
        par2_outputs = power_constrain(par2_outputs, mean2, std2)

        # Built code output table
        code1_output_table = tf.reshape(tf.concat([sys_outputs, par1_outputs], axis=1), (self.trellis_helper.num_states, self.trellis_helper.num_inputs, -1))
        code2_output_table = tf.reshape(tf.concat([sys_outputs, par2_outputs], axis=1), (self.trellis_helper.num_states, self.trellis_helper.num_inputs, -1))
        return code1_output_table, code2_output_table

    # TODO Figure out how to save this config and load up the class from a config
    # def get_config(self):
    #     return {"n_outputs": self.n_outputs, 'window': self.window, 'hidden_layers': self.hidden_layers, 'hidden_dim': self.hidden_dim}

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)


class TurboSystematicSharedEncoder(layers.Layer):

    def __init__(self, shared_encoder, constraint, trellis_helper, tf_interleaver):
        super().__init__()
        assert shared_encoder.n_outputs == 3
        self.shared_encoder = shared_encoder
        self.constraint = constraint
        self.trellis_helper = trellis_helper
        self.tf_interleaver = tf_interleaver
    
    def call(self, input_stream, noisy_channel):
        # Input stream should have size B x block_len
        noninterleaved_input = input_stream
        interleaved_input = tf.gather(noninterleaved_input, self.tf_interleaver["permutation"], axis=1)
        combined_input = tf.concat([noninterleaved_input, interleaved_input], axis=0)
        
        sys, par1, par2 = self.shared_encoder(combined_input)
        non_i_sys = sys[:noninterleaved_input.shape[0]]
        non_i_par = par1[:noninterleaved_input.shape[0]]
        i_par = par2[noninterleaved_input.shape[0]:]

        # Apply the constraint
        constrained_outputs, mean, std = self.constraint(tf.stack([non_i_par, i_par], axis=0))
        non_i_par = constrained_outputs[0]
        i_par = constrained_outputs[1]
        
        # Get the output tables
        code1_output_table, code2_output_table = self.get_code_outputs(mean, std)

        # Corrupt the streams
        non_i_sys_corrupted = noisy_channel(non_i_sys)
        non_i_par_corrupted = noisy_channel(non_i_par)
        i_par_corrupted = noisy_channel(i_par)
        i_sys_corrupted = tf.gather(non_i_sys_corrupted, self.tf_interleaver["permutation"], axis=1)

        # DEBUG Do not corrupt the streams
        # non_i_sys_corrupted = non_i_sys
        # non_i_par_corrupted = non_i_par
        # i_par_corrupted = i_par
        # i_sys_corrupted = tf.gather(non_i_sys_corrupted, self.tf_interleaver["permutation"], axis=1)

        # Ship the completed package to the decoder
        msg = {
            "stream1": non_i_sys_corrupted, 
            "stream2": non_i_par_corrupted, 
            "stream1_i": i_sys_corrupted, 
            "stream3": i_par_corrupted,
            "interleaver": self.tf_interleaver,
            "trellis1": {
                "next_states": self.trellis_helper.tf_next_states,
                "previous_states": self.trellis_helper.tf_prev_states,
                "code_outputs": code1_output_table,
            },
            "trellis2": {
                "next_states": self.trellis_helper.tf_next_states,
                "previous_states": self.trellis_helper.tf_prev_states,
                "code_outputs": code2_output_table,
            },
        }

        return msg

    def get_code_outputs(self, mean, std):
        # Each code output is B x 1 arranged corresponding to possible_inputs
        sys_outputs, par1_outputs, par2_outputs = self.shared_encoder(self.trellis_helper.tf_possible_inputs, pad=False)
        par1_outputs = power_constrain(par1_outputs, mean, std)
        par2_outputs = power_constrain(par2_outputs, mean, std)

        # Built code output table
        code1_output_table = tf.reshape(tf.concat([sys_outputs, par1_outputs], axis=1), (self.trellis_helper.num_states, self.trellis_helper.num_inputs, -1))
        code2_output_table = tf.reshape(tf.concat([sys_outputs, par2_outputs], axis=1), (self.trellis_helper.num_states, self.trellis_helper.num_inputs, -1))
        return code1_output_table, code2_output_table

    # TODO Figure out how to save this config and load up the class from a config
    # def get_config(self):
    #     return {"n_outputs": self.n_outputs, 'window': self.window, 'hidden_layers': self.hidden_layers, 'hidden_dim': self.hidden_dim}

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)


class CNNTrellisHelper(object):
    def __init__(self, window):
        self.window = window
        self.memory = self.window - 1
        self.k = 1
        self.num_inputs = 2 ** self.k
        self.num_states = 2 ** (self.memory)

        # generate all possible binary inputs
        possible_inputs = np.array(list(it.product([0., 1.], repeat=self.window)))
        source_decs = bitarray2dec(possible_inputs[:, 0:self.k], axis=1)
        state_decs = bitarray2dec(possible_inputs[:, self.k:], axis=1)

        shaped_possible_inputs = np.empty((self.num_states, self.num_inputs, self.window), dtype=np.float32)
        shaped_possible_inputs[state_decs, source_decs] = possible_inputs
        
        # Canonicalize the order of possible_inputs
        possible_inputs = shaped_possible_inputs.reshape((-1, self.window))
        source_decs = bitarray2dec(possible_inputs[:, 0:self.k], axis=1)
        state_decs = bitarray2dec(possible_inputs[:, self.k:], axis=1)

        next_state_decs = bitarray2dec(possible_inputs[:, 0:-self.k], axis=1)
        self.next_states = np.empty((self.num_states, self.num_inputs), dtype=np.int32)
        self.next_states[state_decs, source_decs] = next_state_decs

        drop_decs = bitarray2dec(possible_inputs[:, -self.k:], axis=1)
        self.prev_states = np.empty((self.num_states, self.num_inputs, 2), dtype=np.int32)
        self.prev_states[next_state_decs, drop_decs, 0] = state_decs
        input_decs = bitarray2dec(possible_inputs[:, :self.k], axis=1)
        self.prev_states[next_state_decs, drop_decs, 1] = input_decs 

        # Shape is B x L. It is encoder's responsibility to reshpae and recenter as necessary
        self.tf_possible_inputs = tf.constant(possible_inputs)
        self.tf_next_states = tf.constant(self.next_states)
        self.tf_prev_states = tf.constant(self.prev_states)

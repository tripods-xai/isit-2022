import numpy as np

from commpy.utilities import dec2bitarray, bitarray2dec

class BaseModifiedTrellis:
    def __init__(self, generator, bias, delay=0):
        assert generator.ndim == 2, "Generator must be 2-dimensional"
        assert bias.ndim == 1
        assert len(bias) == generator.shape[0]
        self.k = 1
        self.n = generator.shape[0]
        self.total_memory = generator.shape[1] - 1
        self.number_states = 2 ** self.total_memory
        self.number_inputs = 2 ** self.k

        self.next_state_table = np.zeros((self.number_states, self.number_inputs), dtype=int)
        self.output_table = np.zeros((self.number_states, self.number_inputs), dtype=int)

        self.code_type = 'default'
        
        self.generator = generator
        self.bias = bias
        self.delay = delay
    
    @staticmethod
    def shift(shift_register, input_bits):
        if len(shift_register) == 0:
            return shift_register
        else:
            return np.concatenate([input_bits, shift_register[:-len(input_bits)]], axis=0)
    

class ModifiedTrellis(BaseModifiedTrellis):
    """
    To work with Commpy we need to support the following API:

    Attributes
    ----------
    k : int
        Size of the smallest block of input bits that can be encoded using
        the convolutional code.
    n : int
        Size of the smallest block of output bits generated using
        the convolutional code.
    total_memory : int
        Total number of delay elements needed to implement the convolutional
        encoder.
    number_states : int
        Number of states in the convolutional code trellis.
    number_inputs : int
        Number of branches from each state in the convolutional code trellis.
    next_state_table : 2D ndarray of ints
        Table representing the state transition matrix of the
        convolutional code trellis. Rows represent current states and
        columns represent current inputs in decimal. Elements represent the
        corresponding next states in decimal.
    output_table : 2D ndarray of ints
        Table representing the output matrix of the convolutional code trellis.
        Rows represent current states and columns represent current inputs in
        decimal. Elements represent corresponding outputs in decimal.
    code_type : {'default', 'rsc'}, optional
        Use 'rsc' to generate a recursive systematic convolutional code.
        If 'rsc' is specified, then the first 'k x k' sub-matrix of
        G(D) must represent a identity matrix along with a non-zero
        feedback polynomial.
        *Default* is 'default'

    Inputs
    ------
    generator : n x (memory + 1) array
        Binary matrix that for which each row tells us how to make that output bit
        from memory and input bit

    How does the memory and shift work?
    1)  The shift tells us where in the memory the current bit lands. A shift of 
        1 means the "input" bit is the bit after our "current timestep" bit. We don't output the 
        "current timestep" output until shift many steps after the corresponding bit was in our input.
        It can be viewed as a delay on our output and I could code it as such.
    """
    def __init__(self, generator, bias, delay=0):
        super().__init__(generator, bias, delay=delay)
        
        for input_dec in range(self.number_inputs):
            for current_state in range(self.number_states):
                input_bits = dec2bitarray(input_dec, self.k)
                # print(f"Input: {input_bits}")
                shift_register = dec2bitarray(current_state, self.total_memory)
                # print(f"Memory: {shift_register}")
                
                x = np.concatenate([input_bits, shift_register], axis=0)
                output_bits = (np.dot(self.generator, x) + self.bias) % 2
                # print(f"Output bits: {output_bits}")

                output = bitarray2dec(output_bits)
                next_shift_register = self.shift(shift_register, input_bits)
                assert next_shift_register.shape == (self.total_memory,)
                next_state = bitarray2dec(next_shift_register)

                self.output_table[current_state, input_dec] = output
                self.next_state_table[current_state, input_dec] = next_state
    
    def to_rsc(self):
        return RscModifiedTrellis(self.generator, self.bias, delay=self.delay)


class RscModifiedTrellis(BaseModifiedTrellis):
    
    def __init__(self, generator, bias, delay=0):
        super().__init__(generator, bias, delay=delay)
        assert self.n >= 2, "We must have at least two generators to use an rsc"
        self.recursive_generator = generator[:self.k]
        self.recursive_bias = bias[:self.k]

        self.nonsys_generator = generator[self.k:]
        self.nonsys_bias = bias[self.k:]

        self.code_type = 'rsc'

        for input_dec in range(self.number_inputs):
            for current_state in range(self.number_states):
                input_bits = dec2bitarray(input_dec, self.k)
                # print(f"Input: {input_bits}")
                shift_register = dec2bitarray(current_state, self.total_memory)
                # print(f"Memory: {shift_register}")
                
                output_bits = np.zeros(self.n, dtype=int)
                # Systematic stream
                output_bits[:self.k] = input_bits
                # Compute recursive bits
                recursive_input_bits = np.concatenate([input_bits, shift_register], axis=0)
                recursive_bits = (np.dot(self.recursive_generator, recursive_input_bits) + self.recursive_bias) % 2    
                # Compute remaining output bits
                nonsys_input_bits = np.concatenate([recursive_bits, shift_register], axis=0)
                output_bits[self.k:] = (np.dot(self.nonsys_generator, nonsys_input_bits) + self.nonsys_bias) % 2
                # print(f"Output bits: {output_bits}")

                output = bitarray2dec(output_bits)
                next_shift_register = self.shift(shift_register, recursive_bits)
                assert next_shift_register.shape == (self.total_memory,)
                next_state = bitarray2dec(next_shift_register)

                self.output_table[current_state, input_dec] = output
                self.next_state_table[current_state, input_dec] = next_state
    


def modified_conv_encode(msg_bits, trellis, terminate=True):
    assert msg_bits.ndim == 1
    assert len(msg_bits) % trellis.k == 0
    assert msg_bits.dtype == int
    shift = trellis.delay

    shift_inbits = msg_bits[:shift]
    # Don't add additional termination bits if we're already covered from the shift
    term_len = max(trellis.total_memory - shift, 0) if terminate else 0
    modified_msg_bits = np.concatenate([msg_bits[shift:], np.zeros(shift, dtype=int), np.zeros(term_len, dtype=int)], axis=0)
    # print(f"[MCC] Modfied msg bits: {modified_msg_bits}")

    num_inbits = len(modified_msg_bits)
    num_outbits = int(num_inbits / trellis.k * trellis.n)
    # print(f"[MCC] Num inbits {num_inbits}")
    # print(f"[MCC] Num outbits {num_outbits}")

    outbits = np.zeros(num_outbits, dtype=int)

    state = 0
    # Run shift through
    for i in range(shift):
        state = trellis.next_state_table[state, shift_inbits[i]]
        # print(f"[MCC] Shift {i} state {state}")

    for i in range(num_inbits):
        inbit = modified_msg_bits[i]
        # Makes the assumption that k=1
        out_dec = trellis.output_table[state, inbit]
        outbits[i*trellis.n:(i+1)*trellis.n] = dec2bitarray(out_dec, trellis.n)
        # print(f"[MCC] Time {i} current state {state}")
        # print(f"[MCC] Time {i} output val {out_dec}")

        state = trellis.next_state_table[state, inbit]

        
    
    if terminate:
        assert state == 0

    return outbits




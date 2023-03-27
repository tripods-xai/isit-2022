from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from src.utils import sigma2snr




@dataclass
class StateTransitionGraph:
    next_states: tf.Tensor
    # RaggedTensor of |States| x |PrevStates| x 2. Last dimension is pair previous state and transition input
    previous_states: tf.RaggedTensor

    def __post_init__(self):
        tf.debugging.assert_type(self.next_states, tf_type=tf.int32)
        assert self.previous_states.dtype == tf.int32
        assert self.previous_states.shape[0] == self.num_states
    
    @property
    def num_states(self):
        return self.next_states.shape[0]
    
    @property
    def num_inputs(self):
        return self.next_states.shape[1]

    @staticmethod
    def from_next_states(next_states) -> 'StateTransitionGraph':
        num_states = next_states.shape[0]
        num_inputs = next_states.shape[1]
        previous_states_accum: List[List[List[int]]] = [[] for _ in range(num_states)]
        for state in range(num_states):
            for input_sym in range(num_inputs):
                next_state = next_states[state, input_sym]
                previous_states_accum[next_state].append([state, input_sym])
        
        previous_states = tf.ragged.constant(previous_states_accum, inner_shape=(2,))

        return StateTransitionGraph(next_states=next_states, previous_states=previous_states)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, StateTransitionGraph):
            return tf.reduce_all(self.next_states == other.next_states) and tf.reduce_all(self.previous_states == other.previous_states)
        else:
            return NotImplemented

@dataclass
class Trellis:
    state_transitions: StateTransitionGraph
    output_table: tf.Tensor

    def __post_init__(self):
        tf.debugging.assert_type(self.output_table, tf_type=tf.float32)
        assert self.output_table.shape[0] == self.num_states
        assert self.output_table.shape[1] == self.num_inputs


    @property
    def num_outputs(self):
        return self.output_table.shape[2]
    
    @property
    def num_inputs(self):
        return self.state_transitions.num_inputs
    
    @property
    def num_states(self):
        return self.state_transitions.num_states
    
    @property
    def next_states(self):
        return self.state_transitions.next_states
    
    def concat(self, trellis2: 'Trellis') -> 'Trellis':
        if self._check_state_table_compatibility(trellis2):
            return Trellis(
                state_transitions=self.state_transitions, 
                output_table=tf.concat([self.output_table, trellis2.output_table], axis=2)
            )
        else:
            raise ValueError('Input trellis is not compatible with source trellis')
    
    def with_systematic(self) -> 'Trellis':
        np_output_table = np.zeros((self.output_table.shape[0], self.output_table.shape[1], 1))
        np_output_table[:, 1] = 1
        id_output_table = tf.constant(np_output_table, dtype=self.output_table.dtype)
        return Trellis(self.state_transitions, id_output_table).concat(self)

    def _check_state_table_compatibility(self, trellis2: 'Trellis') -> bool:
        return tf.reduce_all(tf.equal(self.state_transitions.next_states, trellis2.state_transitions.next_states))
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Trellis):
            return self.state_transitions == other.state_transitions and tf.reduce_all(self.output_table == other.output_table)
        else:
            return NotImplemented
    
    def __mul__(self, other):
        return Trellis(self.state_transitions, self.output_table * other)
    
    def __add__(self, other):
        return Trellis(self.state_transitions, self.output_table + other)
    
    def __sub__(self, other):
        return Trellis(self.state_transitions, self.output_table - other)
    
    def __truediv__(self, other):
        return Trellis(self.state_transitions, self.output_table / other)
    
    def training(self) -> 'Trellis':
        return Trellis(self.state_transitions, tf.Variable(self.output_table)) # type: ignore

@dataclass
class CodeSettings:
    pass

@dataclass
class UnknownCodeSettings(CodeSettings):
    name: str

@dataclass
class LambdaCodeSettings(CodeSettings):
    function_name: str
    name: str = 'LambdaCode'

@dataclass
class ConcatCodeSettings(CodeSettings):
    codes: List[CodeSettings]
    name: str = 'ConcatCode'

@dataclass
class ComposeCodeSettings(CodeSettings):
    codes: List[CodeSettings]
    name: str = 'ComposeCode'

@dataclass
class IdentityCodeSettings(CodeSettings):
    name: str = 'IdentityCode'

@dataclass
class ProjectionCodeSettings(CodeSettings):
    projection: Tuple[int]
    name: str = 'ProjectionCode'

@dataclass
class TrellisEncoderSettings(CodeSettings):
    trellis: Trellis
    num_states: int
    num_inputs: int
    num_input_channels: int
    num_output_channels: int
    normalize_output_table: bool
    name: str = 'TrellisEncoder'

@dataclass
class ChannelSettings(CodeSettings):
    pass

@dataclass
class UnknownChannelSettings(UnknownCodeSettings, ChannelSettings):
    pass

@dataclass
class AWGNSettings(ChannelSettings):
    sigma: float
    snr: float
    name: str = 'AWGN'

    @staticmethod
    def from_sigma(sigma: float, name: str):
        return AWGNSettings(sigma, sigma2snr(sigma), name=name)

@dataclass
class AdditiveTonAWGNSettings(ChannelSettings):
    sigma: float
    snr: float
    v: float
    name: str = 'AdditiveTonAWGN'

    @staticmethod
    def from_sigma(sigma: float, v=3.0, name: str = 'AdditiveTonAWGN'):
        return AdditiveTonAWGNSettings(sigma=sigma, snr=sigma2snr(sigma), v=v, name=name)

@dataclass
class NonIIDMarkovianGaussianAsAWGNSettings(ChannelSettings):
    sigma:float
    good_sigma: float
    bad_sigma: float
    snr: float
    good_snr: float
    bad_snr: float
    p_gb: float
    p_bg: float
    block_len: int
    name: str

@dataclass
class InterleaverSettings(CodeSettings):
    pass
@dataclass
class FixedPermuteInterleaverSettings(InterleaverSettings):
    permutation: tf.Tensor
    block_len: int
    name: str = 'FixedPermuteInterleaver'

@dataclass
class RandomPermuteInterleaverSettings(InterleaverSettings):
    block_len: int
    name: str = 'RandomPermuteInterleaver'

@dataclass
class DecoderSettings(CodeSettings):
    pass

@dataclass
class BCJRDecoderSettings(DecoderSettings):
    # trellis: Trellis
    trellis_code: TrellisEncoderSettings
    channel: ChannelSettings
    use_max: bool
    num_input_channels: int
    num_output_channels: int
    name: str = 'BCJRDecoder'

@dataclass
class TurboDecoderSettings(DecoderSettings):
    decoder1: DecoderSettings
    decoder2: DecoderSettings
    interleaver: InterleaverSettings
    num_iter: int
    num_noninterleaved_streams: int
    num_input_channels: int
    num_output_channels: int
    name: str = 'TurboDecoder'

@dataclass
class HazzysTurboDecoderSettings(DecoderSettings):
    decoder1: DecoderSettings
    decoder2: DecoderSettings
    interleaver: InterleaverSettings
    num_iter: int
    num_noninterleaved_streams: int
    num_input_channels: int
    num_output_channels: int
    name: str = 'HazzysTurboDecoder'

@dataclass
class EncoderDecoderSettings(CodeSettings):
    pass

@dataclass
class TurboSystematicEncoderDecoderSettings(EncoderDecoderSettings): 
    systematic_code: TrellisEncoderSettings
    interleaved_code: TrellisEncoderSettings
    interleaver: InterleaverSettings
    channel: ChannelSettings
    decoder: TurboDecoderSettings
    rate: Tuple[int, int]
    block_len: int
    use_max: bool
    num_iter: int
    name: str = 'TurboSystematicEncoderDecoder'

@dataclass
class TurboNonsystematicEncoderDecoderSettings(EncoderDecoderSettings): 
    noninterleaved_code: TrellisEncoderSettings
    interleaved_code: TrellisEncoderSettings
    interleaver: InterleaverSettings
    channel: ChannelSettings
    decoder: TurboDecoderSettings
    rate: Tuple[int, int]
    block_len: int
    use_max: bool
    num_iter: int
    name: str = 'TurboNonsystematicEncoderDecoderSettings'
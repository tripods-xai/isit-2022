from enum import Enum
from typing import Callable, Dict
from .channelcoding.channels import AWGN, AdditiveTonAWGN, NonIIDMarkovianGaussianAsAWGN, scale_constraint, Channel

def awgn(sigma, **kwargs) -> AWGN:
    return AWGN(sigma, power_constraint=scale_constraint)

def atn(sigma, **kwargs) -> AdditiveTonAWGN:
    return AdditiveTonAWGN(sigma, v=3, power_constraint=scale_constraint)

def markov_awgn(sigma, block_len, **kwargs) -> NonIIDMarkovianGaussianAsAWGN:
    return NonIIDMarkovianGaussianAsAWGN(sigma, block_len, p_gb=0.8, p_bg=0.8, power_constraint=scale_constraint)

class NoisyChannels(Enum):
    AWGN = 'awgn'
    ATN = 'atn'
    MARKOV_AWGN = 'markov-awgn'

    def __str__(self):
        return self.value

CHANNELS: Dict[Enum, Callable[..., Channel]] = {
    NoisyChannels.AWGN: awgn,
    NoisyChannels.ATN: atn,
    NoisyChannels.MARKOV_AWGN: markov_awgn,
}
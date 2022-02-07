from dataclasses import dataclass
from typing import Dict

from .channelcoding.encoders import TrellisCode

@dataclass
class SystematicTurboEncoderSpec:
    systematic_code: TrellisCode
    interleaved_code: TrellisCode

@dataclass
class NonsystematicTurboEncoderSpec:
    noninterleaved_code: TrellisCode
    interleaved_code: TrellisCode

@dataclass
class TrainerSettings:
    model_id: str
    loss: str
    optimizer: Dict
    block_len: int
    batch_size: int
    write_results_to_log: bool
    logdir: str
    tzname: str
    write_to_tensorboard: bool
    tensorboard_dir: str

@dataclass
class ValidatorSettings:
    model_id: str
    block_len: int
    batch_size: int
    write_results_to_log: bool
    logdir: str
    tzname: str
    write_to_tensorboard: bool
    tensorboard_dir: str
    verbose: int
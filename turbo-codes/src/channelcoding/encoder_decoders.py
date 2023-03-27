import abc
from typing import Tuple, List, cast, Dict, Callable

import tensorflow as tf


from .dataclasses import (
    TurboNonsystematicEncoderDecoderSettings, TurboSystematicEncoderDecoderSettings)
from .bcjr import (BCJRDecoder, SystematicTurboRepeater, TurboDecoder)
from .channels import Channel
from .codes import Code, ComposeCode, ProjectionCode
from .encoders import TrellisCode
from .interleavers import RandomPermuteInterleaver
from .metrics import bit_error_metrics, cross_entropy_with_logits

class EncoderDecoder(Code):

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._metric_values: Dict[str, tf.Tensor] = {}

    @property
    @abc.abstractmethod
    def rate(self) -> Tuple[int, int]:
        pass

    def and_then(self, code2: 'Code') -> 'EncoderDecoder':
        return ComposeEncoderDecoder([self, code2])

    def update_metric(self, metric_dict: Dict[str, tf.Tensor]):
        self._metric_values.update(metric_dict)
    
    @property
    def metric_values(self) -> Dict[str, tf.Tensor]:
        return self._metric_values


class ComposeEncoderDecoder(EncoderDecoder, ComposeCode):

    def __init__(self, codes: List[Code]):
        super(EncoderDecoder, self).__init__(codes)
        assert len(self.codes) > 0
        assert isinstance(self.codes[0], EncoderDecoder)
    
    @property
    def rate(self) -> Tuple[int, int]:
        return cast(EncoderDecoder, self.codes[0]).rate


class TurboSystematicEncoderDecoder(EncoderDecoder):

    def __init__(
        self, 
        systematic_code: TrellisCode, 
        interleaved_code: TrellisCode, 
        channel: Channel, 
        decoder_factory: Callable[..., TurboDecoder],
        block_len: int, 
        use_max: bool = False, 
        num_iter: int = 6,
        interleaver = None,
        name="TurboSystematicEncoderDecoder"
    ):
        super().__init__(name)
        self.systematic_code = systematic_code
        # self.interleaved_code = interleaved_code
        self.interleaved_code_with_systematic = interleaved_code.with_systematic()
        self.channel = channel
        self.block_len = block_len
        self.use_max = use_max
        self.num_iter = num_iter
        if interleaver is None: 
            self.interleaver = RandomPermuteInterleaver(block_len)
        else:
            self.interleaver = interleaver

        # [Sys, Straight_1,..., Straight_{n-1}, Interleaved_1, ..., Interleaved_{m-1}]
        self.encoder = self.systematic_code \
            .concat(
                self.interleaver.and_then(self.interleaved_code_with_systematic).and_then(ProjectionCode((1,)))
            )
        
        self.non_interleaved_bcjr = BCJRDecoder(
            # systematic_code.trellis,
            self.systematic_code,
            self.channel, use_max=use_max
        )
        # interleaved_code_with_systematic = interleaved_code.with_systematic()
        # assert self.interleaved_code_with_systematic.normalize_output_table
        self.interleaved_bcjr = BCJRDecoder(
            # interleaved_code.trellis.with_systematic(),
            self.interleaved_code_with_systematic,
            self.channel, use_max=use_max
        )

        self.repeater = SystematicTurboRepeater(
            num_noninterleaved_streams=self.non_interleaved_bcjr.num_input_channels, 
            interleaver=self.interleaver
        )
        self.decoder = decoder_factory(
            decoder1=self.non_interleaved_bcjr,
            decoder2=self.interleaved_bcjr,
            interleaver=self.interleaver,
            num_iter=num_iter
        )
        
        
    @property
    def rate(self) -> Tuple[int, int]:
        return (self.encoder.num_input_channels, self.encoder.num_output_channels)
    
    @property
    def num_input_channels(self):
        return self.encoder.num_input_channels
    
    @property
    def num_output_channels(self):
        return self.decoder.num_output_channels

    
    def call(self, msg):
        self.reset()
        # # encoded = self.encoder(msg)
        # systematic_encoded = self.systematic_code(msg)
        # # systematic_encoded = tf.Variable(systematic_encoded, name='systematic_encoded')
        # # test = tf.Variable(systematic_encoded * 2 + 1, name='test')
        # # test = test + 1
        # interleaved_encoded = self.interleaved_code(self.interleaver(msg))
        # # interleaved_encoded = tf.Variable(interleaved_encoded, name='interleaved_encoded')
        # encoded = tf.concat([systematic_encoded, interleaved_encoded], axis=2)
        # # encoded = tf.Variable(encoded, name='encoded')
        encoded = self.encoder(msg)
        self.update_metric({'encoded': cast(tf.Tensor, encoded)})
        noisy_encoded = self.channel(encoded)
        # noisy_encoded = tf.Variable(noisy_encoded)
        repeated_noisy_encoded = self.repeater(noisy_encoded)
        # repeated_noisy_encoded = tf.Variable(repeated_noisy_encoded)
        decoded_confidence = self.decoder(repeated_noisy_encoded)
        self.update_metric(bit_error_metrics(msg, decoded_confidence))
        self.update_metric(cross_entropy_with_logits(msg, decoded_confidence))
        # self.update_metric(bit_error_metrics(msg, 2. * msg - 1.))
        return decoded_confidence
    
    def reset(self):
        self.interleaver.reset()
        self._metric_values = {}
        # self.interleaved_bcjr = BCJRDecoder(
        #     self.interleaved_code.with_systematic(),
        #     self.channel, use_max=self.interleaved_bcjr.use_max
        # )
        # self.non_interleaved_bcjr = BCJRDecoder(
        #     # systematic_code.trellis,
        #     self.systematic_code,
        #     self.channel, use_max=self.non_interleaved_bcjr.use_max
        # )
    
    def settings(self) -> TurboSystematicEncoderDecoderSettings:
        return TurboSystematicEncoderDecoderSettings(
            systematic_code=self.systematic_code.settings(),
            interleaved_code=self.interleaved_code_with_systematic.settings(),
            interleaver=self.interleaver.settings(),
            channel=self.channel.settings(),
            decoder=self.decoder.settings(),
            rate=self.rate,
            block_len=self.block_len,
            use_max=self.use_max,
            num_iter=self.num_iter,
            name=self.name
        )
    
    def training(self):
        self.encoder.training()
        # self.systematic_code.training()
        # self.interleaved_code.training()
        # self.interleaver.training()
        self.decoder.training()
        self.channel.training()
    
    def validating(self):
        self.encoder.validating()
        # self.systematic_code.validating()
        # self.interleaved_code.validating()
        # self.interleaver.validating()
        self.decoder.validating()
        self.channel.validating()
    
    def parameters(self) -> List[tf.Variable]:
        return self.encoder.parameters() + self.decoder.parameters()
        # return self.systematic_code.parameters() + self.interleaved_code.parameters() + self.decoder.parameters()

class TurboNonsystematicEncoderDecoder(EncoderDecoder):

    def __init__(
        self, 
        noninterleaved_code: TrellisCode, 
        interleaved_code: TrellisCode, 
        channel: Channel, 
        decoder_factory: Callable[..., TurboDecoder],
        block_len: int, 
        use_max: bool = False, 
        num_iter: int = 10,
        interleaver = None,
        name="TurboNonsystematicEncoderDecoder"
    ):
        super().__init__(name)
        self.noninterleaved_code = noninterleaved_code
        self.interleaved_code = interleaved_code
        self.channel = channel
        self.block_len = block_len
        self.use_max = use_max
        self.num_iter = num_iter
        if interleaver is None: 
            self.interleaver = RandomPermuteInterleaver(block_len)
        else:
            self.interleaver = interleaver

        # [Straight_1,..., Straight_{n}, Interleaved_1, ..., Interleaved_{m}]
        self.encoder = self.noninterleaved_code \
            .concat(
                self.interleaver.and_then(self.interleaved_code)
            )
        
        self.non_interleaved_bcjr = BCJRDecoder(
            self.noninterleaved_code,
            self.channel, use_max=use_max
        )
        self.interleaved_bcjr = BCJRDecoder(
            self.interleaved_code,
            self.channel, use_max=use_max
        )

        self.decoder = decoder_factory(
            decoder1=self.non_interleaved_bcjr,
            decoder2=self.interleaved_bcjr,
            interleaver=self.interleaver,
            num_iter=num_iter
        )

    @property
    def rate(self) -> Tuple[int, int]:
        return (self.encoder.num_input_channels, self.encoder.num_output_channels)
    
    @property
    def num_input_channels(self):
        return self.encoder.num_input_channels
    
    @property
    def num_output_channels(self):
        return self.decoder.num_output_channels

    
    def call(self, msg):
        self.reset()

        encoded = self.encoder(msg)
        self.update_metric({'encoded': cast(tf.Tensor, encoded)})
        noisy_encoded = self.channel(encoded)
        decoded_confidence = self.decoder(noisy_encoded)
        self.update_metric(bit_error_metrics(msg, decoded_confidence))
        self.update_metric(cross_entropy_with_logits(msg, decoded_confidence))
        return decoded_confidence
    
    def reset(self):
        self.interleaver.reset()
        self._metric_values = {}
    
    def training(self):
        self.encoder.training()
        # self.systematic_code.training()
        # self.interleaved_code.training()
        # self.interleaver.training()
        self.decoder.training()
        self.channel.training()
    
    def validating(self):
        self.encoder.validating()
        # self.systematic_code.validating()
        # self.interleaved_code.validating()
        # self.interleaver.validating()
        self.decoder.validating()
        self.channel.validating()
    
    def parameters(self) -> List[tf.Variable]:
        return self.encoder.parameters() + self.decoder.parameters()
    
    def settings(self) -> TurboNonsystematicEncoderDecoderSettings:
        return TurboNonsystematicEncoderDecoderSettings(
            noninterleaved_code=self.noninterleaved_code.settings(),
            interleaved_code=self.interleaved_code.settings(),
            interleaver=self.interleaver.settings(),
            channel=self.channel.settings(),
            decoder=self.decoder.settings(),
            rate=self.rate,
            block_len=self.block_len,
            use_max=self.use_max,
            num_iter=self.num_iter,
            name=self.name
        )

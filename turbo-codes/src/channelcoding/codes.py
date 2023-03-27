import abc
from typing import List, Tuple

import tensorflow as tf
from src.channelcoding.dataclasses import CodeSettings, ComposeCodeSettings, ConcatCodeSettings, IdentityCodeSettings, LambdaCodeSettings, ProjectionCodeSettings, UnknownCodeSettings




class Code(abc.ABC):
    def __init__(self, name: str) -> None:
        # self.name = '_'.join([name, str(uuid.uuid4())[:8]])
        self.name = name
    
    def __call__(self, msg):
        return self.call(msg)
    
    @abc.abstractmethod
    def call(self, msg):
        """
        For a rate n/k code, takes a batch of messages of length t with k channels
        and outputs a batch of messages of length t with n channels.
        """
        pass
    
    @property
    @abc.abstractmethod
    def num_input_channels(self):
        pass

    @property
    @abc.abstractmethod
    def num_output_channels(self):
        pass

    def validate(self):
        pass
    
    def concat(self, code2: 'Code') -> 'Code':
        return ConcatCode([self, code2])
    
    def and_then(self, code2: 'Code') -> 'Code':
        return ComposeCode([self, code2])
    
    # With systematic adds a systematic stream as the first channel
    def with_systematic(self) -> 'Code':
        return ConcatCode([IdentityCode(), self])
    
    def select(self, channels=Tuple[int]):
        return self.and_then(ProjectionCode(channels))
    
    def __mul__(self, other: object):
        def multiply_with(msg):
            return msg * other
        return self.and_then(Lambda(multiply_with))
    
    def __add__(self, other: object):
        def add_with(msg):
            return msg + other
        return self.and_then(Lambda(add_with))
    
    def __sub__(self, other: object):
        def subtract_with(msg):
            return msg - other
        return self.and_then(Lambda(subtract_with))
    
    def __truediv__(self, other: object):
        def divide_with(msg):
            return msg / other
        return self.and_then(Lambda(divide_with))
        
    def settings(self) -> CodeSettings:
        return UnknownCodeSettings(name=self.name)
    
    def reset(self):
        pass
    
    def training(self):
        pass
    
    def validating(self):
        pass
    
    def parameters(self) -> List[tf.Variable]:
        return []

class Lambda(Code):

    def __init__(self, func):
        super().__init__('_'.join(["Lambda", self.func.__name__]))
        self.func = func
        self.function_name = self.func.__name__
    
    @property
    def num_input_channels(self):
        return None
    
    @property
    def num_output_channels(self):
        return None
    
    def call(self, msg):
        return self.func(msg) 
    
    def settings(self) -> LambdaCodeSettings:
        return LambdaCodeSettings(function_name=self.function_name, name=self.name)

class ConcatCode(Code):
    def __init__(self, codes: List[Code], name: str = 'ConcatCode'):
        super().__init__(name)
        self.codes = codes
        self.validate()

    def validate(self):
        non_null_input_channels = [code.num_input_channels for code in self.codes if code.num_input_channels is not None]
        if len(non_null_input_channels) > 0:
            assert all(non_null_input_channels[0] == c for c in non_null_input_channels)
        super().validate()
    
    @property
    def num_input_channels(self):
        non_null_input_channels = [code.num_input_channels for code in self.codes if code.num_input_channels is not None]
        if len(non_null_input_channels) > 0:
            return non_null_input_channels[0]
        else:
            return None        
    
    @property
    def num_output_channels(self):
        output_channels = [code.num_output_channels for code in self.codes]
        if None in output_channels:
            return None
        else:
            return sum(output_channels)
    
    def call(self, msg):
        concat_msg = tf.concat([code(msg) for code in self.codes], axis=2)
        return concat_msg
    
    def reset(self):
        for code in self.codes:
            code.reset()
    
    def training(self):
        for code in self.codes:
            code.training()
    
    def validating(self):
        for code in self.codes:
            code.validating()
    
    def concat(self, code2: 'Code') -> 'Code':
        return ConcatCode(self.codes + [code2], name=self.name)
    
    def settings(self) -> ConcatCodeSettings:
        return ConcatCodeSettings([code.settings() for code in self.codes])
    
    def parameters(self) -> List[tf.Variable]:
        return [p for code in self.codes for p in code.parameters()]


class ComposeCode(Code):
    def __init__(self, codes: List[Code], name='ComposeCode'):
        super().__init__(name)
        self.codes = codes
        self.num_codes = len(codes)

        self.validate()
    
    def validate(self):
        for code1, code2 in zip(self.codes[1:], self.codes[:-1]):
            if code1.num_input_channels is not None and code2.num_output_channels is not None:
                assert code1.num_input_channels == code2.num_output_channels
        super().validate()

    @property
    def num_input_channels(self):
        return self.codes[0].num_input_channels
    
    @property
    def num_output_channels(self):
        return self.codes[-1].num_output_channels
    
    def call(self, msg):
        x = msg
        for code in self.codes:
            x = code(x)
        return x
    
    def reset(self):
        for code in self.codes:
            code.reset()
    
    def training(self):
        for code in self.codes:
            code.training()
    
    def validating(self):
        for code in self.codes:
            code.validating()
    
    def and_then(self, code2: 'Code') -> 'Code':
        return ComposeCode(self.codes + [code2], name=self.name)
    
    def settings(self) -> ComposeCodeSettings:
        return ComposeCodeSettings(codes=[code.settings() for code in self.codes])
    
    def parameters(self) -> List[tf.Variable]:
        return [p for code in self.codes for p in code.parameters()]

class IdentityCode(Code):

    def __init__(self, name: str = 'IdentityCode') -> None:
        super().__init__(name)

    @property
    def num_input_channels(self):
        return None
    
    @property
    def num_output_channels(self):
        return None
    
    def call(self, msg):
        return msg 
    
    def settings(self) -> IdentityCodeSettings:
        return IdentityCodeSettings(name=self.name)


class ProjectionCode(Code):
    def __init__(self, projection: Tuple[int], name: str = 'ProjectionCode'):
        super().__init__(name)
        self.projection = tf.constant(projection)
    
    @property
    def num_input_channels(self):
        return None
    
    @property
    def num_output_channels(self):
        return len(self.projection)
        
    def call(self, msg):
        return tf.gather(msg, self.projection, axis=2)
    
    def settings(self) -> ProjectionCodeSettings:
        return ProjectionCodeSettings(projection=self.projection, name=self.name)

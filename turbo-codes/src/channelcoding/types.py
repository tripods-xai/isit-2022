from typing import Callable, NewType, TypeVar
import tensorflow as tf
from tensor_annotations import tensorflow as ttf
from tensor_annotations import axes

Time = axes.Time
Batch = axes.Batch
Width = axes.Width
Channels = axes.Channels
CodeInputs = NewType('CodeInputs', axes.Axis)
States = NewType('States', axes.Axis)
Input = NewType('Input', axes.Axis)
NextStates = NewType('NextStates', axes.Axis)
PrevStates = NewType('PrevStates', axes.Axis)
InChannels = NewType('InChannels', axes.Axis)
OutChannels = NewType('OutChannels', axes.Axis)

A0 = TypeVar('A0', bound=axes.Axis)
A1 = TypeVar('A1', bound=axes.Axis)
A2 = TypeVar('A2', bound=axes.Axis)
A3 = TypeVar('A3', bound=axes.Axis)
A4 = TypeVar('A4', bound=axes.Axis)
A5 = TypeVar('A5', bound=axes.Axis)
A6 = TypeVar('A6', bound=axes.Axis)
A7 = TypeVar('A7', bound=axes.Axis)
A8 = TypeVar('A8', bound=axes.Axis)

Metric = Callable[[ttf.Tensor3[Batch, Time, Channels], ttf.Tensor3[Batch, Time, Channels]], tf.Tensor]
Loss = Callable[[ttf.Tensor3[Batch, Time, Channels], ttf.Tensor3[Batch, Time, Channels]], tf.Tensor]
from typing import TypeVar, Literal, overload

import tensorflow as tf
from tensor_annotations import axes
from tensor_annotations import tensorflow as ttf

from .types import CodeInputs, Width

A0 = TypeVar('A0', bound=axes.Axis)
A1 = TypeVar('A1', bound=axes.Axis)
A2 = TypeVar('A2', bound=axes.Axis)
A3 = TypeVar('A3', bound=axes.Axis)
A4 = TypeVar('A4', bound=axes.Axis)
A5 = TypeVar('A5', bound=axes.Axis)

Lminus1 = Literal[-1]
L0 = Literal[0]
L1 = Literal[1]
L2 = Literal[2]
L3 = Literal[3]
L4 = Literal[4]

@overload
def dec2bitarray(arr: ttf.Tensor1[A0], num_bits: int, little_endian: bool=...) -> ttf.Tensor2[A0, A1]: ...
@overload
def dec2bitarray(arr: ttf.Tensor2[A0, A1], num_bits: int, little_endian: bool=...) -> ttf.Tensor3[A0, A1, A2]: ...
def dec2bitarray(arr, num_bits:int, little_endian: bool=False):
    if little_endian:
        shift_arr = tf.range(num_bits)
    else:
        shift_arr = tf.range(num_bits)[::-1]
    return tf.bitwise.right_shift(tf.expand_dims(arr, -1), shift_arr) % 2

def enumerate_binary_inputs(window: int) -> ttf.Tensor2[CodeInputs, Width]:
    return dec2bitarray(tf.range(2 ** window), window)

def safe_int(x: float) -> int:
    assert x.is_integer()
    return int(x)

def base_2_accumulator(length: int, little_endian: bool=False) -> ttf.Tensor1[Width]:
    powers_of_2 = tf.bitwise.left_shift(1, tf.range(length))
    if little_endian:
        return powers_of_2
    else:
        return powers_of_2[::-1]

@overload
def bitarray2dec(arr: ttf.Tensor3[A0, A1, A2], little_endian: bool=..., axis: Lminus1=...) -> ttf.Tensor2[A0, A1]: ...
@overload
def bitarray2dec(arr: ttf.Tensor3[A0, A1, A2], little_endian: bool=..., axis: L0=...) -> ttf.Tensor2[A1, A2]: ...
@overload
def bitarray2dec(arr: ttf.Tensor3[A0, A1, A2], little_endian: bool=..., axis: L1=...) -> ttf.Tensor2[A0, A2]: ...
@overload
def bitarray2dec(arr: ttf.Tensor3[A0, A1, A2], little_endian: bool=..., axis: L2=...) -> ttf.Tensor2[A0, A1]: ...
def bitarray2dec(arr, little_endian=False, axis=-1):
    base_2 = base_2_accumulator(arr.shape[axis], little_endian=little_endian)
    return tf.tensordot(arr, base_2, axes=[[axis], [0]])

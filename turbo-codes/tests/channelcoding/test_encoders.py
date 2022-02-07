import numpy as np
import tensorflow as tf
import pytest

from src.channelcoding.dataclasses import StateTransitionGraph, Trellis
from src.channelcoding.encoders import AffineConvolutionalCode, GeneralizedConvolutionalCode, TrellisCode

from commpy import channelcoding as cc

from src.channelcoding.interleavers import PermuteInterleaver
from src.codes import turboae_binary_exact_nonsys

from .utils import interleaver_to_commpy, vsystematic_turbo_encode


def test_trellis_code_construct_basic():
    next_states = tf.constant(
        [[1, 2],
         [0, 1],
         [1, 2]], dtype=tf.int32
    )
    state_transitions = StateTransitionGraph.from_next_states(next_states)
    output_table = tf.constant(
        [[[-0.12057394,  2.1424615 ,  0.98776376], [-0.03558801, -0.9341358 , -0.78986055]],
         [[ 1.1309315 ,  1.2084178 , -1.0634259 ], [-0.44598797,  1.0388252 ,  0.03602624]],
         [[ 0.28023955,  0.6486982 ,  1.1790744 ], [ 1.8008167 ,  1.3370351 , -0.1548117 ]]], dtype=tf.float32
    )
    trellis = Trellis(state_transitions, output_table)

    code = TrellisCode(trellis)
    assert code.trellis == trellis
    assert code.num_inputs == next_states.shape[1]
    assert code.num_states == next_states.shape[0]
    assert code.num_input_channels == 1
    assert code.num_output_channels == 3


def test_trellis_code_call_basic():
    next_states = tf.constant(
        [[1, 2],
         [0, 1],
         [1, 2]], dtype=tf.int32
    )
    state_transitions = StateTransitionGraph.from_next_states(next_states)
    output_table = tf.constant(
        [[[-0.12057394,  2.1424615 ,  0.98776376], [-0.03558801, -0.9341358 , -0.78986055]],
         [[ 1.1309315 ,  1.2084178 , -1.0634259 ], [-0.44598797,  1.0388252 ,  0.03602624]],
         [[ 0.28023955,  0.6486982 ,  1.1790744 ], [ 1.8008167 ,  1.3370351 , -0.1548117 ]]], dtype=tf.float32
    )
    trellis = Trellis(state_transitions, output_table)

    code = TrellisCode(trellis)    
    msg = tf.constant([
        [[1],[0],[1],[1]],
        [[0],[1],[1],[0]],
        [[0],[0],[1],[1]]
    ], dtype=tf.int32)
    expected_output = tf.constant([
        [[-0.03558801, -0.9341358 , -0.78986055],[ 0.28023955,  0.6486982 ,  1.1790744 ],[-0.44598797,  1.0388252 ,  0.03602624],[-0.44598797,  1.0388252 ,  0.03602624]],
        [[-0.12057394,  2.1424615 ,  0.98776376],[-0.44598797,  1.0388252 ,  0.03602624],[-0.44598797,  1.0388252 ,  0.03602624],[ 1.1309315 ,  1.2084178 , -1.0634259 ]],
        [[-0.12057394,  2.1424615 ,  0.98776376],[ 1.1309315 ,  1.2084178 , -1.0634259 ],[-0.03558801, -0.9341358 , -0.78986055],[ 1.8008167 ,  1.3370351 , -0.1548117 ]]
    ], dtype=tf.float32)

    assert tf.reduce_all(code(msg) == expected_output)

def test_gen_conv_code_construct_no_feedback_basic():
    table = tf.constant([
        [1, 1],
        [1, 1],
        [2, 0],
        [0, 0],
        [2, 1],
        [2, 1],
        [2, 0],
        [2, 2]], dtype=tf.float32)
    code = GeneralizedConvolutionalCode(table)
    assert tf.reduce_all(code.table == table)
    assert code.feedback is None
    assert code.num_input_channels == 1
    assert code.num_output_channels == 2
    assert code.num_possible_windows == 8
    assert code.window == 3

    expected_next_states = tf.constant([
        [0, 1],
        [2, 3],
        [0, 1],
        [2, 3]
    ], dtype=tf.int32)
    expected_output_table = tf.constant([
        [[1, 1], [1, 1]],
        [[2, 0], [0, 0]],
        [[2, 1], [2, 1]],
        [[2, 0], [2, 2]]
    ], dtype=tf.float32)
    assert code.trellis == Trellis(StateTransitionGraph.from_next_states(expected_next_states), expected_output_table)

def test_gen_conv_code_call_no_feedback_basic():
    table = tf.constant([
        [1, 1],
        [1, 1],
        [2, 0],
        [0, 0],
        [2, 1],
        [2, 1],
        [2, 0],
        [2, 2]], dtype=tf.float32)
    code = GeneralizedConvolutionalCode(table)
    msg = tf.constant([
        [[1],[0],[1],[1]],
        [[0],[1],[1],[0]],
        [[0],[0],[1],[1]]
    ], dtype=tf.int32)
    expected_output = tf.constant([
        [[1, 1],[2, 0],[2, 1],[0, 0]],
        [[1, 1],[1, 1],[0, 0],[2, 0]],
        [[1, 1],[1, 1],[1, 1],[0, 0]]
    ], dtype=tf.float32)

    assert tf.reduce_all(code(msg) == expected_output)

def test_gen_conv_code_call_trellis_equiv():
    table = tf.constant([
        [1, 1],
        [1, 1],
        [2, 0],
        [0, 0],
        [2, 1],
        [2, 1],
        [2, 0],
        [2, 2]], dtype=tf.float32)
    code = GeneralizedConvolutionalCode(table)
    msg = tf.cast(tf.random.uniform((100, 100, 1), dtype=tf.int32, maxval=2), dtype=tf.float32)
    
    trellis_code = TrellisCode(code.trellis)

    assert tf.reduce_all(code(msg) == trellis_code(msg))

    table2 = tf.constant([
        [1],
        [1],
        [2],
        [0],
        [2],
        [2],
        [2],
        [2]], dtype=tf.float32)
    code2 = GeneralizedConvolutionalCode(table2)
    msg2 = tf.random.uniform((100, 100, 1), dtype=tf.int32, maxval=2)
    
    trellis_code2 = TrellisCode(code2.trellis)

    assert tf.reduce_all(code2(msg2) == trellis_code2(msg2))

def test_affine_conv_code_construct_basic():
    gen = tf.constant([[1, 0, 1], [1, 1, 1]])
    bias = tf.constant([1, 0])
    code = AffineConvolutionalCode(gen, bias)

    expected_table = tf.constant([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [0, 1],
        [1, 0],
        [0, 0],
        [1, 1]], dtype=tf.float32)

    assert code.num_input_channels == 1
    assert code.num_output_channels == 2
    assert tf.reduce_all(code.bias == bias)
    assert tf.reduce_all(code.generator == gen)
    assert tf.reduce_all(code.table == expected_table) 

def test_affine_conv_code_call_basic():
    gen = tf.constant([[1, 0, 1], [1, 1, 1]])
    bias = tf.constant([1, 0])
    code = AffineConvolutionalCode(gen, bias)
    msg = tf.constant([
        [[1],[0],[1],[1]],
        [[0],[1],[1],[0]],
        [[0],[0],[1],[1]]
    ], dtype=tf.int32)
    expected_output = tf.constant([
        [[0, 1],[1, 1],[1, 0],[0, 0]],
        [[1, 0],[0, 1],[0, 0],[0, 0]],
        [[1, 0],[1, 0],[0, 1],[0, 0]]
    ], dtype=tf.float32)

    assert tf.reduce_all(code(msg) == expected_output)

def test_gen_conv_code_construct_with_feedback_basic():
    table = tf.constant([
        [1, 1],
        [1, 1],
        [2, 0],
        [0, 0],
        [2, 1],
        [2, 1],
        [2, 0],
        [2, 2]], dtype=tf.float32)
    feedback = tf.constant([1, 1, 1, 0, 0, 0, 1, 1], dtype=tf.int32)
    code = GeneralizedConvolutionalCode(table, feedback=feedback)
    assert tf.reduce_all(code.table == table)
    assert tf.reduce_all(code.feedback == feedback)
    assert code.num_input_channels == 1
    assert code.num_output_channels == 2
    assert code.num_possible_windows == 8
    assert code.window == 3

    expected_next_states = tf.constant([
        [1, 1],
        [3, 2],
        [0, 0],
        [3, 3]
    ], dtype=tf.int32)
    expected_output_table = tf.constant([
        [[1, 1], [1, 1]],
        [[0, 0], [2, 0]],
        [[2, 1], [2, 1]],
        [[2, 2], [2, 2]]
    ], dtype=tf.float32)
    assert code.trellis == Trellis(StateTransitionGraph.from_next_states(expected_next_states), expected_output_table)

def test_gen_conv_code_call_with_feedback_basic():
    table = tf.constant([
        [1, 1],
        [1, 1],
        [2, 0],
        [0, 0],
        [2, 1],
        [2, 1],
        [2, 0],
        [2, 2]], dtype=tf.float32)
    feedback = tf.constant([1, 1, 1, 0, 0, 0, 1, 1], dtype=tf.int32)
    code = GeneralizedConvolutionalCode(table, feedback)
    msg = tf.constant([
        [[1],[0],[1],[1]],
        [[0],[1],[1],[0]],
        [[0],[0],[1],[1]]
    ], dtype=tf.int32)
    expected_output = tf.constant([
        [[1, 1],[0, 0],[2, 2],[2, 2]],
        [[1, 1],[2, 0],[2, 1],[1, 1]],
        [[1, 1],[0, 0],[2, 2],[2, 2]]
    ], dtype=tf.float32)

    assert tf.reduce_all(code(msg) == expected_output)

def test_affine_nonsys_rsc_equivalence():
    gen = tf.constant([[1, 0, 1], [1, 1, 1]])
    bias = tf.constant([1, 0])
    nonsys_code = AffineConvolutionalCode(gen, bias)
    rsc_code = nonsys_code.to_rsc()
    msg = tf.constant([
        [[1],[0],[1],[1]],
        [[0],[1],[1],[0]],
        [[0],[0],[1],[1]]
    ], dtype=tf.int32)

    nonsys_out = nonsys_code(msg)
    rsc_out = rsc_code(nonsys_out[:, :, 0:1])
    
    tf.reduce_all(nonsys_out == rsc_out)

def test_gen_conv_nonsys_rsc_equivalence():
    table = tf.constant([
        [1, 1],
        [0, 1],
        [0, 3],
        [1, 0],
        [0, 1],
        [1, 2],
        [1, 0],
        [0, 2]], dtype=tf.float32)
    nonsys_code = GeneralizedConvolutionalCode(table, feedback=None)
    rsc_code = nonsys_code.to_rsc()
    msg = tf.constant([
        [[1],[0],[1],[1]],
        [[0],[1],[1],[0]],
        [[0],[0],[1],[1]]
    ], dtype=tf.int32)

    nonsys_out = nonsys_code(msg)
    rsc_out = rsc_code(nonsys_out[:, :, 0:1])
    
    tf.reduce_all(nonsys_out == rsc_out)

def test_gen_conv_nonsys_rc_conversion_exceptions():
    table_not_bin = tf.constant([
        [1, 1],
        [1, 1],
        [2, 0],
        [0, 0],
        [2, 1],
        [2, 1],
        [2, 0],
        [2, 2]], dtype=tf.float32)
    table_not_invert = tf.constant([
        [1, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 2]], dtype=tf.float32)
    feedback = tf.constant([1, 1, 1, 0, 0, 0, 1, 1], dtype=tf.int32)
    table_good = tf.constant([
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 2]], dtype=tf.float32)
    feedback = tf.constant([1, 1, 1, 0, 0, 0, 1, 1], dtype=tf.int32)

    with pytest.raises(ValueError) as e_info:
        GeneralizedConvolutionalCode(table_not_bin, feedback=None).to_rc()
    with pytest.raises(ValueError) as e_info:
        GeneralizedConvolutionalCode(table_not_invert, feedback=None).to_rc()
    with pytest.raises(ValueError) as e_info:
        GeneralizedConvolutionalCode(table_good, feedback=feedback).to_rc()

    GeneralizedConvolutionalCode(table_good, feedback=None).to_rc()

def test_compare_757_with_commpy():
    tf_code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rsc()
    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)

    msg = tf.random.uniform((100, 100, 1), dtype=tf.int32, maxval=2)
    np_msg = msg[:, :, 0].numpy()
    
    tf_out = tf_code(msg)
    commpy_out = np.apply_along_axis(cc.conv_encode, axis=1, arr=np_msg, trellis=commpy_trellis, termination='cont')

    np.testing.assert_array_equal(tf_out.numpy().reshape((100, 200)), commpy_out)

def test_compare_turbo_757_with_commpy():
    block_len = 100
    tf_code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rc()
    tf_interleaver = PermuteInterleaver(block_len)
    tf_encoder = tf_code.with_systematic() \
        .concat(tf_interleaver.and_then(tf_code))

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    commpy_interleaver = interleaver_to_commpy(tf_interleaver)

    msg = tf.random.uniform((100, block_len, 1), dtype=tf.int32, maxval=2)
    np_msg = msg.numpy()
    
    tf_out = tf_encoder(msg)
    commpy_out = vsystematic_turbo_encode(np_msg, commpy_trellis, commpy_trellis, commpy_interleaver)

    np.testing.assert_array_equal(tf_out.numpy(), commpy_out)

def test_compare_turbo_755_0_with_commpy():
    block_len = 100
    tf_code1 = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0]))
    tf_code2 = AffineConvolutionalCode(tf.constant([[1, 0, 1]]), tf.constant([0]))
    tf_interleaver = PermuteInterleaver(block_len)
    tf_encoder = tf_code1 \
        .concat(tf_interleaver.and_then(tf_code2))

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    commpy_interleaver = interleaver_to_commpy(tf_interleaver)

    msg = tf.random.uniform((100, block_len, 1), dtype=tf.int32, maxval=2)
    np_msg = msg.numpy()
    
    tf_out = tf_encoder(msg)
    commpy_out = vsystematic_turbo_encode(np_msg, commpy_trellis, commpy_trellis, commpy_interleaver)

    np.testing.assert_array_equal(tf_out.numpy(), commpy_out)

def test_turboae_exact_nonsys_trellis_compare():
    block_len = 100
    encoder_spec = turboae_binary_exact_nonsys()
    tf_interleaver = PermuteInterleaver(block_len)

    encoder = encoder_spec.noninterleaved_code \
            .concat(
                tf_interleaver.and_then(encoder_spec.interleaved_code)
            )
    
    trellis_noninterleaved_code = TrellisCode(encoder_spec.noninterleaved_code.trellis)
    trellis_interleaved_code = TrellisCode(encoder_spec.interleaved_code.trellis)

    trellis_encoder = trellis_noninterleaved_code \
            .concat(
                tf_interleaver.and_then(trellis_interleaved_code)
            )
    
    msg = tf.random.uniform((1000, block_len, 1), dtype=tf.int32, maxval=2)
    assert tf.reduce_all(encoder(msg) == trellis_encoder(msg))
import tensorflow as tf
import pytest

from src.channelcoding.dataclasses import StateTransitionGraph, Trellis

def test_construct_state_transitions_basic():
    next_states = tf.constant(
        [[1, 2],
         [0, 1],
         [1, 2]], dtype=tf.int32
    )
    prev_states = tf.ragged.constant([
        [[1, 0]],
        [[0, 0], [1, 1], [2, 0]],
        [[0, 1], [2, 1]]
    ], inner_shape=(2,))

    state_transitions = StateTransitionGraph(next_states, prev_states)
    assert tf.reduce_all(state_transitions.next_states == next_states)
    assert tf.reduce_all(state_transitions.previous_states == prev_states)

def test_eq_state_transition_graph_basic():
    next_states = tf.constant(
        [[1, 2],
         [0, 1],
         [1, 2]], dtype=tf.int32
    )
    prev_states = tf.ragged.constant([
        [[1, 0]],
        [[0, 0], [1, 1], [2, 0]],
        [[0, 1], [2, 1]]
    ], inner_shape=(2,))

    wrong_next_states = tf.constant(
        [[0, 2],
         [0, 1],
         [1, 2]], dtype=tf.int32
    )
    wrong_prev_states = tf.ragged.constant([
        [[1, 0]],
        [[0, 0], [1, 1], [2, 0]],
        [[0, 4], [2, 1]]
    ], inner_shape=(2,))

    good_state_transitions = StateTransitionGraph(next_states, prev_states)
    good_state_transitions_other = StateTransitionGraph(next_states, prev_states)
    bad_state_transitions1 = StateTransitionGraph(wrong_next_states, prev_states)
    bad_state_transitions2 = StateTransitionGraph(next_states, wrong_prev_states)
    bad_state_transitions3 = StateTransitionGraph(wrong_next_states, wrong_prev_states)

    assert good_state_transitions == good_state_transitions_other
    assert bad_state_transitions1 != good_state_transitions_other
    assert good_state_transitions != bad_state_transitions1
    assert bad_state_transitions2 != good_state_transitions_other
    assert good_state_transitions != bad_state_transitions2
    assert bad_state_transitions3 != good_state_transitions_other
    assert good_state_transitions != bad_state_transitions3


def test_from_next_states_basic():
    next_states = tf.constant(
        [[1, 2],
         [0, 1],
         [1, 2]], dtype=tf.int32
    )
    expected_prev_states = tf.ragged.constant([
        [[1, 0]],
        [[0, 0], [1, 1], [2, 0]],
        [[0, 1], [2, 1]]
    ], inner_shape=(2,))

    state_transitions = StateTransitionGraph.from_next_states(next_states)

    assert state_transitions == StateTransitionGraph(next_states, expected_prev_states)


def test_construct_trellis_basic():
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
    assert trellis.state_transitions == state_transitions
    assert tf.reduce_all(trellis.output_table == output_table)
    assert trellis.num_outputs == 3


def test_trellis_eq_basic():
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
    wrong_next_states = tf.constant(
        [[1, 2],
         [0, 1],
         [1, 0]], dtype=tf.int32
    )
    wrong_state_transitions = StateTransitionGraph.from_next_states(wrong_next_states)
    wrong_output_table = tf.constant(
        [[[-0.12057394,  2.1424615 ,  0.98776376], [-0.03558801, -0.9341358 , -0.78986055]],
         [[ 1.1309315 ,  1.2084178 , -1.0634259 ], [-0.44598797,  1.0388252 ,  0.03602624]],
         [[ 0.28023955,  0.6486982 ,  27849 ], [ 1.8008167 ,  1.3370351 , -0.1548117 ]]], dtype=tf.float32
    )

    good_trellis = Trellis(state_transitions, output_table)
    good_trellis_other = Trellis(state_transitions, output_table)
    bad_trellis1 = Trellis(wrong_state_transitions, output_table) 
    bad_trellis2 = Trellis(state_transitions, wrong_output_table)
    bad_trellis3 = Trellis(wrong_state_transitions, wrong_output_table)

    assert good_trellis == good_trellis_other
    assert bad_trellis1 != good_trellis_other
    assert good_trellis != bad_trellis1
    assert bad_trellis2 != good_trellis_other
    assert good_trellis != bad_trellis2
    assert bad_trellis3 != good_trellis_other
    assert good_trellis != bad_trellis3


def test_trellis_concat_basic():
    next_states = tf.constant(
        [[1, 2],
         [0, 1],
         [1, 2]], dtype=tf.int32
    )
    state_transitions = StateTransitionGraph.from_next_states(next_states)
    incompat_next_states = tf.constant(
        [[1, 2],
         [0, 0],
         [1, 2]], dtype=tf.int32
    )
    incompat_state_transitions = StateTransitionGraph.from_next_states(incompat_next_states)
    output_table = tf.constant(
        [[[-0.12057394,  2.1424615 ,  0.98776376], [-0.03558801, -0.9341358 , -0.78986055]],
         [[ 1.1309315 ,  1.2084178 , -1.0634259 ], [-0.44598797,  1.0388252 ,  0.03602624]],
         [[ 0.28023955,  0.6486982 ,  1.1790744 ], [ 1.8008167 ,  1.3370351 , -0.1548117 ]]], dtype=tf.float32)
    output_table_other = tf.constant([
        [[ 1.2826903 ,  0.29455426, -0.4418655 ], [-1.2639806 , -1.9767599 ,  0.47084045]],
        [[ 0.01317857,  0.1059294 ,  0.18517183], [-0.8374467 ,  0.06358027,  0.35246328]],
        [[ 0.6101968 ,  0.64533025,  2.2886977 ], [ 0.9031279 ,  0.0716385 , -0.6904388 ]]], dtype=tf.float32)
    output_table_concat = tf.constant(
        [[[-0.12057394,  2.1424615 ,  0.98776376,  1.2826903 ,  0.29455426, -0.4418655 ], [-0.03558801, -0.9341358 , -0.78986055, -1.2639806 , -1.9767599 ,  0.47084045]],
         [[ 1.1309315 ,  1.2084178 , -1.0634259,  0.01317857,  0.1059294 ,  0.18517183 ], [-0.44598797,  1.0388252 ,  0.03602624, -0.8374467 ,  0.06358027,  0.35246328]],
         [[ 0.28023955,  0.6486982 ,  1.1790744,  0.6101968 ,  0.64533025,  2.2886977  ], [ 1.8008167 ,  1.3370351 , -0.1548117,  0.9031279 ,  0.0716385 , -0.6904388  ]]], dtype=tf.float32)
    
    trellis = Trellis(state_transitions, output_table)
    trellis_other = Trellis(state_transitions, output_table_other)
    trellis_incompat = Trellis(incompat_state_transitions, output_table_other)

    assert trellis.concat(trellis_other) == Trellis(state_transitions, output_table_concat)
    with pytest.raises(ValueError):
        trellis.concat(trellis_incompat)
    
def test_trellis_with_systematic_basic():
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
    systematic_output_table = tf.constant(
        [[[0., -0.12057394,  2.1424615 ,  0.98776376], [1., -0.03558801, -0.9341358 , -0.78986055]],
         [[0., 1.1309315 ,  1.2084178 , -1.0634259  ], [1., -0.44598797,  1.0388252 ,  0.03602624]],
         [[0., 0.28023955,  0.6486982 ,  1.1790744  ], [1.,  1.8008167 ,  1.3370351 , -0.1548117 ]]], dtype=tf.float32
    )

    trellis = Trellis(state_transitions, output_table)
    assert trellis.with_systematic() == Trellis(state_transitions, systematic_output_table)
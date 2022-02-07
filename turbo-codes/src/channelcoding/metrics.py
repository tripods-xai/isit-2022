import typing
from typing import Dict

from tensor_annotations.axes import Batch, Channels, Time
import tensorflow as tf
from tensor_annotations import tensorflow as ttf

def bit_error_metrics(original_msg: ttf.Tensor3[Batch, Time, Channels], msg_confidence: ttf.Tensor3[Batch, Time, Channels]) -> Dict[str, tf.Tensor]:
    ber_for_block = tf.reduce_mean(tf.cast(tf.not_equal(original_msg, tf.cast(msg_confidence > 0, tf.float32)), tf.float32), axis=[1, 2])
    block_error_for_block = tf.cast(ber_for_block > 0, tf.float32)
    return {'ber': ber_for_block, 'bler': block_error_for_block}

# Loss function
def cross_entropy_with_logits(original_msg: ttf.Tensor3[Batch, Time, Channels], msg_confidence: ttf.Tensor3[Batch, Time, Channels]) -> Dict[str, tf.Tensor]:
    return {'cross_entropy': tf.nn.sigmoid_cross_entropy_with_logits(labels=original_msg, logits=msg_confidence)}
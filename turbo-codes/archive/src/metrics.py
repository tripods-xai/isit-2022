import tensorflow as tf

def bit_error_rate(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0, tf.float32)
    return tf.reduce_mean(tf.cast(tf.not_equal(y_true, y_pred), tf.float32), axis=-1)

def block_error_rate(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0, tf.float32)
    return tf.cast(tf.reduce_any(tf.not_equal(y_true, y_pred), axis=-1), tf.float32)
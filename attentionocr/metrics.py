import tensorflow as tf


def masked_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    y_true_indices = tf.argmax(y_true, axis=-1)
    y_pred_indices = tf.argmax(y_pred, axis=-1)
    padding = tf.equal(y_pred_indices, 0)  # 0 is the magic value for PAD
    mask = 1.0 - tf.cast(padding, dtype=tf.float32)
    correct = tf.cast(tf.equal(y_true_indices, y_pred_indices), dtype=tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)

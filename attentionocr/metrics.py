import tensorflow as tf


def masked_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    y_true_indices = tf.argmax(y_true, axis=-1)
    y_pred_indices = tf.argmax(y_pred, axis=-1)
    padding = tf.equal(y_pred_indices, 0)  # TODO 0 is the magic value for PAD
    mask = 1.0 - tf.cast(padding, dtype=tf.float32)
    correct = tf.cast(tf.equal(y_true_indices, y_pred_indices), dtype=tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)


def fan_loss(y_true, y_pred, attention_weights, attention_target, ratio: float = 0.01) -> tf.Tensor:
        loss = masked_loss(y_true, y_pred)

        indices = tf.argmax(y_true, axis=2)
        attn_char_mask = tf.less_equal(indices, 3)
        attn_char_mask = (1 - tf.cast(attn_char_mask, dtype=tf.float32))
        attn_sample_mask = tf.equal(-1, tf.reduce_mean(attention_target, axis=2))
        attn_sample_mask = (1 - tf.cast(attn_sample_mask, dtype=tf.float32))
        attn_loss = tf.losses.binary_crossentropy(attention_weights, attention_target)
        attn_loss = tf.reduce_sum(attn_loss * attn_char_mask * attn_sample_mask) / tf.reduce_sum(attn_char_mask)

        return (1.0 - ratio) * loss + ratio * attn_loss


def masked_loss(y_true, y_pred) -> tf.Tensor:
    indices = tf.argmax(y_true, axis=2)
    mask = tf.equal(indices, 0)
    mask = (1 - tf.cast(mask, dtype=tf.float32))
    loss = tf.losses.categorical_crossentropy(y_true, y_pred)
    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return loss

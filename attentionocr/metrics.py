import tensorflow as tf


def masked_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    y_true_indices = tf.argmax(y_true, axis=-1)
    y_pred_indices = tf.argmax(y_pred, axis=-1)
    padding = tf.equal(y_pred_indices, 0)  # TODO 0 is the magic value for PAD
    mask = 1.0 - tf.cast(padding, dtype=tf.float32)
    correct = tf.cast(tf.equal(y_true_indices, y_pred_indices), dtype=tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)


def fan_loss(y_true, y_pred, attention_weights, attention_target) -> tf.Tensor:
        indices = tf.argmax(y_true, axis=2)
        mask = tf.equal(indices, 0)
        mask = (1 - tf.cast(mask, dtype=tf.float32))
        loss = tf.losses.categorical_crossentropy(y_true, y_pred)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

        attn_mask = tf.less_equal(indices, 3)
        attn_mask = (1 - tf.cast(attn_mask, dtype=tf.float32))
        attn_loss = tf.losses.categorical_crossentropy(attention_weights, attention_target)
        attn_loss = tf.reduce_sum(attn_loss * attn_mask) / tf.reduce_sum(attn_mask)

        return (loss + attn_loss) / 2.0

        # return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

import numpy as np
import tensorflow as tf

def masked_accuracy(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true.shape) == 3  # batch, letters, vocab
    assert y_true.shape[0] == y_pred.shape[0]
    if y_true.shape[0] is None:
        return 1.0
    accuracy = 0
    for item in range(y_true.shape[0]):
        y, pred = y_true[item], y_pred[item]
        y = tf.argmax(y, axis=-1)
        pred = tf.argmax(pred, axis=-1)
        length = np.where(y == 1)[0][0]
        y = y[:length]
        pred = pred[:length]
        accuracy += tf.reduce_sum(tf.cast(y == pred, tf.float32)) / float(length)
    return (accuracy / float(y_true.shape[0])).numpy()

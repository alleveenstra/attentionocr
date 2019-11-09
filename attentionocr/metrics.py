import numpy as np
import tensorflow as tf

def masked_accuracy(y_true, y_pred) -> float:
    print(tf.__version__)
    print(tf.executing_eagerly())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true.shape) == 3  # batch, letters, vocab
    assert y_true.shape[0] == y_pred.shape[0]
    if y_true.shape[0] is None:
        return 1.0
    accuracy = 0
    for item in range(y_true.shape[0]):
        y, pred = y_true[item], y_pred[item]
        y = np.argmax(y, axis=-1)
        pred = np.argmax(pred, axis=-1)
        longest = max(np.min(np.where(y == 1)), np.min(np.where(pred == 1)))
        y = y[:longest]
        pred = pred[:longest]
        accuracy += (y == pred).sum() / float(longest)
    return accuracy / float(y_true.shape[0])

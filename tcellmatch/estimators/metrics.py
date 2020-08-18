import tensorflow as tf


def custom_r2(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return: (graphs,)
    """
    r2 = 1. - tf.reduce_sum(tf.math.square(y_true - y_pred)) / \
         tf.reduce_sum(tf.math.square(y_true - tf.reduce_mean(y_true)))
    return tf.reduce_mean(r2)


def custom_logr2(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return: (graphs,)
    """
    eps = 1.
    y_true = tf.math.log(y_true + eps)
    y_pred = tf.math.log(y_pred + eps)
    return custom_r2(y_true=y_true, y_pred=y_pred)

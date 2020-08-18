import functools
from typing import List, Tuple

import tensorflow as tf


def build_layer_set(
        layer_types: List[str],
        activations: List[str],
        filter_widths: List[Tuple],
        widths: List[int],
        strides: List[Tuple],
        dropout_rate=0.5,
        dtype=tf.float32
):
    """
    Build a sub-network as a callable function.

    TODO write conv layer stack (conv2d + dropout + pooling) as keras.layers.Layer instance.

    :param layer_types: Type of hidden layers.
    :param activations: Activation functions by hidden layers.
    :param widths: The width of the each hidden layer or number of filters if convolution.
    :param filter_widths: The width of filters if convolutional layer, otherwise ignored.
    :param strides: The strides if convolutional layer, otherwise ignored.
    :param dropout_rate: rate for drop out layer
    :returns net: A callable that maps the input data tensor to the latent space
        as a `tfd.Distribution` instance.
    """
    assert len(layer_types) == len(widths), \
        "supply same number of layer_types and width: number of hidden layers "
    assert len(activations) == len(widths), \
        "supply same number of activations and width: number of hidden layers "
    assert len(filter_widths) == len(widths), \
        "supply same number of activations and width: number of hidden layers "

    # Write handles for keras layer objects which only expose necessary parameters:
    # Using 2D convolutions here as transpose of conv1d is not yet supplied.
    # The forth dimension is only one element long, so not effectively used.
    conv2D = functools.partial(
        tf.keras.layers.Conv2D,
        padding='same',
        data_format='channels_last',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        dtype=dtype
    )

    deconv2D = functools.partial(
        tf.keras.layers.Conv2DTranspose,
        padding='same',
        output_padding=None,
        data_format='channels_last',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        dtype=dtype
    )

    dense = functools.partial(
        tf.keras.layers.Dense,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        dtype=dtype
    )

    # Assemble list of sequential keras layers according to user input.
    net = []
    for i, (l, a, w, fw, s) in enumerate(zip(layer_types, activations, widths, filter_widths, strides)):
        if l.lower() == "conv":
            net.append(conv2D(
                filters=w,
                kernel_size=fw,
                activation=a,
                strides=s
            ))
            if i > 0 and layer_types[i - 1] != "deconv":
                net.append(tf.keras.layers.BatchNormalization(dtype=dtype))
                net.append(tf.keras.layers.Dropout(rate=dropout_rate))

        elif l.lower() == "deconv":
            # Flatten first if last layer was a convolution:
            if i > 0 and layer_types[i - 1] == "conv":
                net.append(tf.keras.layers.Flatten())

            net.append(deconv2D(
                filters=w,
                kernel_size=fw,
                activation=a,
                strides=s
            ))
            net.append(tf.keras.layers.BatchNormalization(dtype=dtype))
            net.append(tf.keras.layers.Dropout(rate=dropout_rate))

        elif l.lower() == "dense":
            # Flatten first if last layer was a convolution:
            if i > 0 and layer_types[i - 1] == "conv":
                net.append(tf.keras.layers.Flatten())

            net.append(dense(
                units=2*w,  # one w for mean of Guassian, the other for stddev
                activation=a
            ))
            net.append(tf.keras.layers.BatchNormalization(dtype=dtype))
            net.append(tf.keras.layers.Dropout(rate=dropout_rate))
        else:
            raise ValueError("layer_type %s not recognized" % l)

    # Build sub-network based on list of layers.
    # net = tf.keras.Sequential(net)
    return net

import functools
import tensorflow as tf


class LayerInception(tf.keras.layers.Layer):
    """ A layer class that implements inception on 1D sequence data.

    Instances of this class can be used as layers in the context of tensorflow Models.
    Uses the following inception-type layout:
    x -> 1x1 -> y_0
    x -> 1x1 -> 3x3 -> y_1
    x -> 1x1 -> 5x5 -> y_2
    x -> 1x1 -> 7x7 -> y_3
    Followed by concatenation of all y_i.
    Note that there are 2 1x1 layers: 1x1 layer 0 and 1, which have different output dimensions.
    """
    sublayer_1x1_level1: tf.keras.layers.Conv1D
    sublayer_3x3_level1: tf.keras.layers.Conv1D
    sublayer_3x3_level2: tf.keras.layers.Conv1D
    sublayer_5x5_level1: tf.keras.layers.Conv1D
    sublayer_5x5_level2: tf.keras.layers.Conv1D
    sublayer_7x7_level1: tf.keras.layers.Conv1D
    sublayer_7x7_level2: tf.keras.layers.Conv1D
    sublayer_pool5x5_level1: tf.keras.layers.MaxPool1D
    sublayer_pool5x5_level2: tf.keras.layers.Conv1D
    sublayer_dropout: tf.keras.layers.Dropout

    def __init__(
            self,
            n_filters_1x1: int,
            n_filters_out: int,
            residual_connection: bool,
            dropout: float = 0,
            input_shape=None,
            feed_forward_size: int = 20,
            dtype=tf.float32
    ):
        """

        :param shape_embedding: Dimensionality of embedding space of each attention head.
        :param residual_connection: Whether to use residual connections across sub-layer blocks.
            See code in the call() method.
        :param dropout: Dropout rate for training.
        :param input_shape: Input shape determination can be delayed by setting this to None.
        :param feed_forward_size:
        """
        tf.keras.layers.Layer.__init__(self=self, dtype=dtype)
        self.n_filters_1x1 = n_filters_1x1
        self.n_filters_out = n_filters_out
        self.residual_connection = residual_connection
        self.dropout_rate = dropout
        self.feed_forward_size = feed_forward_size
        self.input_shapes = input_shape

    def build(self, input_shape):
        """ Defines sub-layers.

        Allow for delayed evaluation of input shapes.
        """
        if self.input_shapes is not None:
            input_shape = self.input_shapes

        conv1D = functools.partial(
            tf.keras.layers.Conv1D,
            filters=self.n_filters_out,
            activation='linear',
            strides=1,
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
            dtype=self.dtype
        )

        self.sublayer_1x1_level1 = conv1D(kernel_size=1)
        self.sublayer_3x3_level1 = conv1D(kernel_size=1)
        self.sublayer_3x3_level2 = conv1D(kernel_size=3)
        self.sublayer_5x5_level1 = conv1D(kernel_size=1)
        self.sublayer_5x5_level2 = conv1D(kernel_size=5)
        self.sublayer_7x7_level1 = conv1D(kernel_size=1)
        self.sublayer_7x7_level2 = conv1D(kernel_size=7)
        self.sublayer_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, x, training=True, **kwargs):
        """ Forward pass through layer.

        :param x: input tensor [batch_size, length sequence, number of input features]
        :param training: Whether forward pass is in context of training or prediction: Use drop-out only during
            training.
        :return: output tensor [batch_size, length sequence, shape_embedding]
        """
        # Concatenate output of filters in channel dimension:
        # "Depth concatenation" as proposed by the inception paper.
        y = tf.concat([
            self.sublayer_1x1_level1(x),
            self.sublayer_3x3_level2(self.sublayer_3x3_level1(x)),
            self.sublayer_5x5_level2(self.sublayer_5x5_level1(x)),
            self.sublayer_7x7_level2(self.sublayer_7x7_level1(x))
        ], axis=2)

        if training:
            y = self.sublayer_dropout(y)
        return y

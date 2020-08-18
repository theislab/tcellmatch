import tensorflow as tf
from typing import Union, Tuple


class LayerConv(tf.keras.layers.Layer):
    """ A layer class that implements sequence convolution.

    Instances of this class can be used as layers in the context of tensorflow Models.
    This layer implements convolution and pooling. Uses the following sequence:

    convolution -> batch normalisation -> activation -> drop-out -> pooling
    TODO read a bit into whether this is the best order.
    """
    sublayer_conv: tf.keras.layers.Conv1D
    sublayer_batchnorm: tf.keras.layers.BatchNormalization
    sublayer_act: tf.keras.layers.Activation
    sublayer_dropout: tf.keras.layers.Dropout
    sublayer_pool: tf.keras.layers.MaxPool1D

    def __init__(
            self,
            activation: str,
            filter_width: int,
            filters: int,
            stride: int,
            pool_size: int,
            pool_stride: int,
            batch_norm: bool = True,
            dropout: float = 0.0,
            input_shape: Union[Tuple, None] = None,
            trainable: bool = True,
            dtype=tf.float32
    ):
        """

        Note: Addition of batch normalisation results in non-trainable weights in this layer.

        :param activation: Activation function. Refer to documentation of tf.keras.layers.Conv2D
        :param filter_width: Number of neurons per filter. Refer to documentation of tf.keras.layers.Conv2D
        :param filters: Number of filters / output channels. Refer to documentation of tf.keras.layers.Conv2D
        :param stride: Stride size for convolution on sequence. Refer to documentation of tf.keras.layers.Conv2D
        :param pool_size: Size of max-pooling, ie. number of output nodes to pool over.
            Refer to documentation of tf.keras.layers.MaxPool2D:pool_size
        :param pool_stride: Stride of max-pooling.
            Refer to documentation of tf.keras.layers.MaxPool2D:strides
        :param batch_norm: Whether to perform batch normalization.
        :param dropout: Dropout rate to use during training.
        :param input_shape:
        :param trainable:
        :param dtype:
        """
        tf.keras.layers.Layer.__init__(self=self, trainable=trainable, dtype=dtype)

        self.activation = activation
        self.filter_width = filter_width
        self.filters = filters
        self.stride = stride
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.input_shapes = input_shape
        self.fwd_pass = []

    def build(self, input_shape):
        """ Initialise layers.

        Allows for delayed evaluation of input shapes.
        """
        self.sublayer_conv = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.filter_width,
            activation='linear',
            strides=self.stride if self.stride is not None else None,
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
        if self.batch_norm:
            self.sublayer_batchnorm = tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
                beta_initializer='zeros',
                gamma_initializer='ones',
                moving_mean_initializer='zeros',
                moving_variance_initializer='ones',
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                renorm=False,
                renorm_clipping=None,
                renorm_momentum=0.99,
                fused=None,
                trainable=True,
                virtual_batch_size=None,
                adjustment=None,
                dtype=self.dtype
            )
        self.sublayer_act = tf.keras.layers.Activation(self.activation)
        if self.dropout > 0:
            self.sublayer_dropout = tf.keras.layers.Dropout(rate=self.dropout)
        if self.pool_size is not None:
            self.sublayer_pool = tf.keras.layers.MaxPool1D(
                pool_size=self.pool_size,
                strides=self.pool_stride if self.pool_stride is not None else None,
                padding='same'
            )

    def call(self, x, training=True, **kwargs):
        """ Forward pass through layer.

        :param x: input tensor
        :param training: Whether forward pass is in context of training or prediction: Use drop-out only during
            training.
        :return: output tensor
        """
        x = self.sublayer_conv(x)
        if self.batch_norm:
            x = self.sublayer_batchnorm(x, training=training)
        x = self.sublayer_act(x)
        if self.dropout > 0 and training:
            x = self.sublayer_dropout(x)
        if self.pool_size is not None:
            x = self.sublayer_pool(x)
        return x

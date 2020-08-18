import tensorflow as tf


class LayerAaEmbedding(tf.keras.layers.Layer):
    """ A layer class that implements amino acid embedding.

    Instances of this class can be used as layers in the context of tensorflow Models.
    This layer implements 1x1 convolutions to map a given amino acid embedding (such as one-hot) into a learnable
    new space of choosable dimensionality.

    """
    sublayer_conv2d: tf.keras.layers.Conv2D
    fwd_pass: list

    def __init__(
            self,
            shape_embedding: int,
            squeeze_2D_sequence: bool,
            trainable: bool = True,
            dropout: float = 0.1,
            input_shape=None,
            dtype=tf.float32
    ):
        """

        :param shape_embedding: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if shape_embedding==0.
        :param squeeze_2D_sequence:
        :param trainable:
        :param dropout:
        :param input_shape:
        :param dtype:
        """
        tf.keras.layers.Layer.__init__(self=self, trainable=trainable, name="aa_embedding", dtype=dtype)
        if shape_embedding < 0:
            raise ValueError("aa_embedding_dim has to be >0")

        self.shape_embedding = shape_embedding
        self.squeeze_2D_sequence = squeeze_2D_sequence
        self.dropout = dropout
        self.input_shapes = input_shape
        self.sublayer_conv2d = None

    def build(self, input_shape):
        """ Initialise layers.

        Allows for delayed evaluation of input shapes.
        """
        if self.input_shapes is not None:
            input_shape = self.input_shapes

        if self.shape_embedding is not None:
            if self.shape_embedding == 0:
                # Set to length of amino acid encoding space.
                self.shape_embedding = int(input_shape[-1])

            self.sublayer_conv2d = tf.keras.layers.Conv1D(
                filters=self.shape_embedding,
                kernel_size=1,
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

    def call(self, x, **kwargs):
        """ Forward pass through layer.

        :param x: input tensor [batch_size, length sequence, number of input features]
        :return: output tensor [batch_size, length sequence, shape_embedding]
        """
        if self.shape_embedding is not None:
            x = self.sublayer_conv2d(x)

        return x

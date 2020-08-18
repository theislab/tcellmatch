import tensorflow as tf


class LayerMultiheadSelfAttention(tf.keras.layers.Layer):
    """ Keras layer for multi-headed self-attention."""
    q_embedding: tf.keras.layers.Dense
    k_embedding: tf.keras.layers.Dense
    v_embedding: tf.keras.layers.Dense
    final_dense: tf.keras.layers.Dense
    split_heads_reshape: tf.keras.layers.Reshape
    split_heads_permute: tf.keras.layers.Permute
    merge_heads_reshape: tf.keras.layers.Reshape
    merge_heads_permute: tf.keras.layers.Permute

    def __init__(
            self,
            width_embedding: int,
            n_heads: int,
            residual_connection: bool,
            attention_dropout: float = 0.,
            input_shape=None,
            name="sa",
            dtype=tf.float32,
            **kwargs
    ):
        """

        :param width_embedding: Dimensionality of embedding space of each attention head.
        :param n_heads: Number of attention heads to use (multi-headed attention).
        :param residual_connection: Whether to use residual connections across sub-layer blocks.
            See code in the call() method.
        :param attention_dropout: Dropout rate of QK embedding for training.
        :param input_shape: Input shape determination can be delayed by setting this to None.
        :param feed_forward_size:
        """
        super(LayerMultiheadSelfAttention, self).__init__(name=name, dtype=dtype, **kwargs)
        self.total_width_embedding = int(width_embedding * n_heads)
        self.n_heads = n_heads
        self.width_embedding = width_embedding
        self.qk_dropout = attention_dropout
        self.residual_connection = residual_connection
        self.input_shapes = input_shape
        self.seq_len = None

    def build(self, input_shape):
        """ Defines sub-layers.

        Allow for delayed evaluation of input shapes.
        """
        self.seq_len = input_shape[1]
        # Self-attention computes the relevance of an entry at each position with respect to
        # all other positions and uses 3 tensors to do so: Q, K, V. We use the same notation
        # as in the original transformer paper here.
        # Q: Linear embedding of query.
        # K: Linear embedding of key.
        # V: Linear transform computing value.
        # These three layers constitute sub-layers of the self-attention layer and include multi-headed self-attention
        # in the TODO's dimension of the tensor.
        self.q_embedding = tf.keras.layers.Dense(units=self.total_width_embedding, use_bias=False)
        self.k_embedding = tf.keras.layers.Dense(units=self.total_width_embedding, use_bias=False)
        self.v_embedding = tf.keras.layers.Dense(units=self.total_width_embedding, use_bias=False)

        self.split_heads_reshape = tf.keras.layers.Reshape(
            [self.seq_len, self.n_heads, self.width_embedding], name="reshape_split"
        )
        self.split_heads_permute = tf.keras.layers.Permute([2, 1, 3], name="permute_split")

        self.merge_heads_reshape = tf.keras.layers.Reshape(
            [self.seq_len, self.total_width_embedding], name="reshape_merge"
        )
        self.merge_heads_permute = tf.keras.layers.Permute([2, 1, 3], name="permute_merge")
        
        # Dense layers to compute embedding of self-attention output.
        self.final_dense = tf.keras.layers.Dense(
            units=int(input_shape[-1]),
            activation="relu",
            name="final_dense"
        )

    def call(self, inputs, training=True, **kwargs):
        """ Forward pass through layer.

        Note that the output is similar to a RNN with return_sequence==True,
        ie an embedding is returned for each input position.

        :param inputs: input tensor [batch_size, length sequence, number of input features]
        :param training: Whether forward pass is in context of training or prediction: Use drop-out only during
            training.
        :return: output tensor [batch_size, length sequence, shape_embedding]
        """
        # Project input in linear embedding using Q, K, V. Here, the different
        # attention heads ("channels") are still concatenated.
        q = self.q_embedding(inputs)
        k = self.k_embedding(inputs)
        v = self.v_embedding(inputs)
        # Reshape concatenated q, k, v tensors into tensors with attention heads as a separate dimension.
        # Yields: [batch_size, num_heads, length, hidden_size/num_heads]
        q = self.split_heads_permute(self.split_heads_reshape(q))
        k = self.split_heads_permute(self.split_heads_reshape(k))
        v = self.split_heads_permute(self.split_heads_reshape(v))

        # Scale q based on number of heads so that the output of the attention layer is not as dependent
        # on the number of attention heads. This is used for numeric stability.
        q = q / tf.sqrt(tf.cast(self.n_heads, dtype=q.dtype))
        # Compute self.attention as dot-product attention in the linear embedding:
        qk = tf.matmul(q, k, transpose_b=True)
        qk = tf.keras.layers.Softmax()(qk)
        if training:
            qk = tf.keras.layers.Dropout(self.qk_dropout)(qk)
        qkv = tf.matmul(qk, v)

        # Concatenate attention heads into a single dimension.
        y = self.merge_heads_reshape(self.merge_heads_permute(qkv))  # [batch_size, length, hidden_size]
        # Project into space of the same size as input tensor.
        y = self.final_dense(y)  # [batch_size, length, TODO]

        # Apply residual connection across entire self-attention mechanism:
        if self.residual_connection:
            y = tf.keras.layers.LayerNormalization(center=False, scale=False)(inputs + y)
        return y

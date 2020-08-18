import tensorflow as tf
from typing import List, Union

from tcellmatch.models.layers.layer_aa_embedding import LayerAaEmbedding
from tcellmatch.models.layers.layer_attention import LayerMultiheadSelfAttention
from tcellmatch.models.layers.layer_conv import LayerConv


class ModelBiRnn:

    forward_pass: list
    forward_pass_pep: list
    final_dense: list

    def __init__(
            self,
            labels_dim: int,
            input_shapes: tuple,
            model: str,
            topology: List[int],
            split: bool,
            residual_connection: bool = False,
            aa_embedding_dim: Union[int, None] = 0,
            depth_final_dense: int = 1,
            out_activation: str = "linear",
            dropout: float = 0.0
    ):
        """ BiLSTM-based feed-forward network with optional 1x1 convolutional embedding layer.

        Build the feed forward network as a tf.keras.Model object.
        The 1x1 convolutional embedding layer can be used to create a lower-dimensional embedding of
        one hot encoded amino acids before the sequence model is used.

        :param model: Layer type to use: {"bilstm", "bigru"}.
        :param topology: The depth of each bilstm layer (length of feature vector)
        :param dropout: drop out rate for bilstm.
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param out_activation: Identifier of output activation function, this depends on
            assumption on labels and cost function:

            - "linear" for binding strength data measured as counts
            - "sigmoid" for binary binding events with multiple events per cell
            - "softmax" for binary binding events with one event per cell
        """
        self.args = {
            "labels_dim": labels_dim,
            "input_shapes": input_shapes,
            "model": model,
            "residual_connection": residual_connection,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "out_activation": out_activation,
            "dropout": dropout,
            "split": split
        }
        # Attention!!! output dim of biLSTM is two times of the dim of LSTM TODO this is not accounted for yet.
        # return_sequences=False means we only return the state of last cell in LSTM
        self.labels_dim = labels_dim
        self.input_shapes = input_shapes
        self.model = model
        self.topology = topology
        self.residual_connection = residual_connection
        self.aa_embedding_dim = aa_embedding_dim
        self.depth_final_dense = depth_final_dense
        self.out_activation = out_activation
        self.dropout = dropout
        self.split = split
        self.run_eagerly = False
        self.x_len = input_shapes[4]

        input_tcr = tf.keras.layers.Input(
            shape=(input_shapes[0], input_shapes[1], input_shapes[2]),
            name='input_tcr'
        )
        input_covar = tf.keras.layers.Input(
            shape=(input_shapes[3]),
            name='input_covar'
        )

        # Preprocessing:
        x = input_tcr
        x = tf.squeeze(x, axis=[1])  # squeeze out chain
        x = 2 * (x - 0.5)
        # Optional amino acid embedding:
        if aa_embedding_dim is not None:
            x = LayerAaEmbedding(
                shape_embedding=self.aa_embedding_dim,
                squeeze_2D_sequence=True
            )(x)
        # Split input into tcr and peptide
        if self.split:
            pep = x[:, self.x_len:, :]
            x = x[:, :self.x_len, :]  # TCR sequence from here on.
        for i, w in enumerate(self.topology):
            if self.model.lower() == "bilstm":
                x = tf.keras.layers.Bidirectional(
                    layer=tf.keras.layers.LSTM(
                        units=w,
                        return_sequences=True if i < len(self.topology) - 1 else False,
                        dropout=self.dropout,
                        activation="tanh",
                        use_bias=True,
                        recurrent_dropout=0,
                        recurrent_activation='sigmoid',
                        recurrent_initializer='orthogonal',
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        recurrent_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        unroll=False,
                        unit_forget_bias=True
                    ),
                    merge_mode='concat'
                )(x)
                if self.split:
                    pep = tf.keras.layers.Bidirectional(
                        layer=tf.keras.layers.LSTM(
                            units=w,
                            return_sequences=True if i < len(self.topology) - 1 else False,
                            dropout=self.dropout,
                            activation="tanh",
                            use_bias=True,
                            recurrent_dropout=0,
                            recurrent_activation='sigmoid',
                            recurrent_initializer='orthogonal',
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros',
                            kernel_regularizer=None,
                            recurrent_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            unroll=False,
                            unit_forget_bias=True
                        ),
                        merge_mode='concat'
                    )(pep)
            elif self.model.lower() == "bigru":
                x = tf.keras.layers.Bidirectional(
                    layer=tf.keras.layers.GRU(
                        units=w,
                        return_sequences=True if i < len(self.topology) - 1 else False,
                        dropout=self.dropout,
                        recurrent_dropout=0,
                        activation="tanh",
                        recurrent_activation='sigmoid',
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        recurrent_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        unroll=False,
                        reset_after=True
                    ),
                    merge_mode='concat'
                )(x)
                if self.split:
                    pep = tf.keras.layers.Bidirectional(
                        layer=tf.keras.layers.GRU(
                            units=w,
                            return_sequences=True if i < len(self.topology) - 1 else False,
                            dropout=self.dropout,
                            recurrent_dropout=0,
                            activation="tanh",
                            recurrent_activation='sigmoid',
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal',
                            bias_initializer='zeros',
                            kernel_regularizer=None,
                            recurrent_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            unroll=False,
                            reset_after=True
                        ),
                        merge_mode='concat'
                    )(pep)
        if self.split:
            x = tf.concat([x, pep], axis=1)
        # Optional concatenation of non-sequence covariates.
        if input_covar.shape[1] > 0:
            x = tf.concat([x, input_covar], axis=1)
        # Final dense layers.from
        for i in range(self.depth_final_dense):
            x = tf.keras.layers.Dense(
                units=self.labels_dim,
                activation="relu" if i < self.depth_final_dense - 1 else self.out_activation,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None
            )(x)
        output = x

        self.training_model = tf.keras.models.Model(
            inputs=[input_tcr, input_covar],
            outputs=output,
            name='model_birnn'
        )


class ModelSa:

    forward_pass: list
    final_dense: list

    def __init__(
            self,
            labels_dim: int,
            input_shapes: tuple,
            attention_size: List[int],
            attention_heads: List[int],
            split: bool,
            residual_connection=False,
            aa_embedding_dim: Union[int, None] = 0,
            depth_final_dense: int = 1,
            out_activation: str = "linear",
            dropout: float = 0.0
    ):
        """ Self-attention-based feed-forward network.

        Build the feed forward network as a tf.keras.Model object.

        :param dropout: drop out rate for lstm.
        :param attention_size: hidden size for attention, could be divided by attention_heads.
        :param attention_heads: number of heads in attention.
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param out_activation: Identifier of output activation function, this depends on
            assumption on labels and cost function:

            - "linear" for binding strength data measured as counts
            - "sigmoid" for binary binding events with multiple events per cell
            - "softmax" for binary binding events with one event per cell
        """
        self.args = {
            "labels_dim": labels_dim,
            "input_shapes": input_shapes,
            "attention_size": attention_size,
            "attention_heads": attention_heads,
            "residual_connection": residual_connection,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "out_activation": out_activation,
            "dropout": dropout,
            "split": split
        }
        self.input_shapes = input_shapes
        self.labels_dim = labels_dim
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.residual_connection = residual_connection
        self.aa_embedding_dim = aa_embedding_dim
        self.depth_final_dense = depth_final_dense
        self.out_activation = out_activation
        self.dropout = dropout
        self.split = split
        assert not split, "not implemented"
        self.run_eagerly = False
        self.x_len = input_shapes[4]

        input_tcr = tf.keras.layers.Input(
            shape=(input_shapes[0], input_shapes[1], input_shapes[2]),
            name='input_tcr'
        )
        input_covar = tf.keras.layers.Input(
            shape=(input_shapes[3]),
            name='input_covar'
        )

        # Preprocessing:
        x = input_tcr
        x = tf.squeeze(x, axis=[1])  # squeeze out chain
        x = 2 * (x - 0.5)
        # Optional amino acid embedding:
        if aa_embedding_dim is not None:
            x = LayerAaEmbedding(
                shape_embedding=self.aa_embedding_dim,
                squeeze_2D_sequence=True
            )(x)
        # Self-attention layers.
        for i, (w, n) in enumerate(zip(self.attention_size, self.attention_heads)):
            x = LayerMultiheadSelfAttention(
                width_embedding=w,
                n_heads=n,
                residual_connection=self.residual_connection,
                attention_dropout=self.dropout,
                name="sa_" + str(i)
            )(x)
        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])
        # Optional concatenation of non-sequence covariates.
        if input_covar.shape[1] > 0:
            x = tf.concat([x, input_covar], axis=1)
        # Final dense layers.from
        self.final_dense = []
        for i in range(self.depth_final_dense):
            x = tf.keras.layers.Dense(
                units=self.labels_dim,
                activation="relu" if i < self.depth_final_dense - 1 else self.out_activation.lower(),
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None
            )(x)
        output = x

        self.training_model = tf.keras.models.Model(
            inputs=[input_tcr, input_covar],
            outputs=output,
            name='model_sa'
        )


class ModelConv:

    def __init__(
            self,
            labels_dim: int,
            input_shapes: tuple,
            activations: List[str],
            filters: List[int],
            filter_widths: List[int],
            strides: List[int],
            split: bool,
            pool_sizes: Union[None, List[int]] = None,
            pool_strides: Union[None, List[int]] = None,
            batch_norm: bool = True,
            aa_embedding_dim: Union[int, None] = 0,
            depth_final_dense: int = 1,
            out_activation: str = "linear",
            dropout: float = 0.0
    ):
        """ Convolution-based feed-forward network.

        Build the feed forward network as a tf.keras.Model object.

        :param activations: Activation functions by hidden layers.
            Refer to documentation of tf.keras.layers.Activation
        :param filters: The width of the each hidden layer or number of filters if convolution.
            Refer to documentation of tf.keras.layers.Conv2D
        :param filter_widths: The width of filters if convolutional layer, otherwise ignored.
            Refer to documentation of tf.keras.layers.Conv2D
        :param strides: The strides if convolutional layer, otherwise ignored.
            Refer to documentation of tf.keras.layers.Conv2D
        :param pool_sizes: Size of max-pooling, ie. number of output nodes to pool over, by layer.
            Refer to documentation of tf.keras.layers.MaxPool2D:pool_size
        :param pool_strides: Stride of max-pooling by layer.
            Refer to documentation of tf.keras.layers.MaxPool2D:strides
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param out_activation: Identifier of output activation function, this depends on
            assumption on labels and cost function:

            - "linear" for binding strength data measured as counts
            - "sigmoid" for binary binding events with multiple events per cell
            - "softmax" for binary binding events with one event per cell
        """
        self.args = {
            "labels_dim": labels_dim,
            "input_shapes": input_shapes,
            "aa_embedding_dim": aa_embedding_dim,
            "out_activation": out_activation,
            "activations": activations,
            "filters": filters,
            "filter_widths": filter_widths,
            "strides": strides,
            "pool_strides": pool_strides,
            "pool_sizes": pool_sizes,
            "batch_norm": batch_norm,
            "depth_final_dense": depth_final_dense,
            "dropout": dropout,
            "split": split
        }
        self.labels_dim = labels_dim
        self.input_shapes = input_shapes
        self.aa_embedding_dim = aa_embedding_dim
        self.out_activation = out_activation
        self.activations = activations
        self.filters = filters
        self.filter_widths = filter_widths
        self.strides = strides
        self.pool_strides = pool_strides if pool_strides is not None else [None for x in self.activations]
        self.pool_sizes = pool_sizes if pool_sizes is not None else [None for x in self.activations]
        self.batch_norm = batch_norm
        self.depth_final_dense = depth_final_dense
        self.dropout = dropout
        self.split = split
        assert not split, "not implemented"
        self.run_eagerly = False
        self.x_len = input_shapes[4]

        input_tcr = tf.keras.layers.Input(
            shape=(input_shapes[0], input_shapes[1], input_shapes[2]),
            name='input_tcr'
        )
        input_covar = tf.keras.layers.Input(
            shape=(input_shapes[3]),
            name='input_covar'
        )

        # Preprocessing:
        x = input_tcr
        x = tf.squeeze(x, axis=[1])  # squeeze out chain
        x = 2 * (x - 0.5)
        # Optional amino acid embedding:
        if aa_embedding_dim is not None:
            x = LayerAaEmbedding(
                shape_embedding=self.aa_embedding_dim,
                squeeze_2D_sequence=False
            )(x)
        # Convolutional layers.
        for i, (a, f, fw, s, psize, pstride) in enumerate(zip(
                self.activations,
                self.filters,
                self.filter_widths,
                self.strides,
                self.pool_sizes,
                self.pool_strides,
        )):
            x = LayerConv(
                activation=a,
                filter_width=fw,
                filters=f,
                stride=s,
                pool_size=psize,
                pool_stride=pstride,
                batch_norm=self.batch_norm,
                dropout=self.dropout
            )(x)
        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])
        # Optional concatenation of non-sequence covariates.
        if input_covar.shape[1] > 0:
            x = tf.concat([x, input_covar], axis=1)
        # Final dense layers.from
        for i in range(self.depth_final_dense):
            x = tf.keras.layers.Dense(
                units=self.labels_dim,
                activation="relu" if i < self.depth_final_dense - 1 else self.out_activation,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None
            )(x)
        output = x

        self.training_model = tf.keras.models.Model(
            inputs=[input_tcr, input_covar],
            outputs=output,
            name='model_conv'
        )


class ModelLinear:

    def __init__(
            self,
            labels_dim: int,
            input_shapes: tuple,
            aa_embedding_dim: Union[int, None] = 0,
            out_activation: str = "linear"
    ):
        """ Linear densely conected feed-forward network.

        Build the feed forward network as a tf.keras.Model object.

        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param out_activation: Identifier of output activation function, this depends on
            assumption on labels and cost function:

            - "linear" for binding strength data measured as counts
            - "sigmoid" for binary binding events with multiple events per cell
            - "softmax" for binary binding events with one event per cell
        """
        self.args = {
            "labels_dim": labels_dim,
            "aa_embedding_dim": aa_embedding_dim,
            "out_activation": out_activation,
            "input_shapes": input_shapes
        }
        self.aa_embedding_dim = aa_embedding_dim
        self.labels_dim = labels_dim
        self.out_activation = out_activation
        self.run_eagerly = False
        self.x_len = input_shapes[4]

        input_tcr = tf.keras.layers.Input(
            shape=(input_shapes[0], input_shapes[1], input_shapes[2]),
            name='input_tcr'
        )
        input_covar = tf.keras.layers.Input(
            shape=(input_shapes[3]),
            name='input_covar'
        )

        x = input_tcr
        x = tf.squeeze(x, axis=[1])  # squeeze out chain
        x = 2 * (x - 0.5)
        # Amino acid embedding layer.
        if aa_embedding_dim is not None:
            x = LayerAaEmbedding(
                shape_embedding=self.aa_embedding_dim,
                squeeze_2D_sequence=False
            )(x)
        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])
        if input_covar.shape[1] > 0:
            x = tf.concat([x, input_covar], axis=1)

        # Linear layer.
        output = tf.keras.layers.Dense(
            units=self.labels_dim,
            activation=out_activation,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None
        )(x)

        self.training_model = tf.keras.models.Model(
            inputs=[input_tcr, input_covar],
            outputs=output,
            name='model_linear'
        )


class ModelNoseq:

    forward_pass: List[tf.keras.layers.Dense]

    def __init__(
            self,
            labels_dim: int,
            input_shapes: tuple,
            out_activation: str = "linear",
            n_layers: int = 3
    ):
        """ Densely conected feed-forward network that does not include sequence data.

        Build the feed forward network as a tf.keras.Model object.

        :param out_activation: Identifier of output activation function, this depends on
            assumption on labels and cost function:

            - "linear" for binding strength data measured as counts
            - "sigmoid" for binary binding events with multiple events per cell
            - "softmax" for binary binding events with one event per cell
        """
        self.args = {
            "labels_dim": labels_dim,
            "out_activation": out_activation,
            "input_shapes": input_shapes,
            "n_layers": n_layers
        }
        self.labels_dim = labels_dim
        self.out_activation = out_activation
        self.n_layers = n_layers
        self.run_eagerly = False
        self.x_len = input_shapes[4]

        input_tcr = tf.keras.layers.Input(
            shape=(input_shapes[0], input_shapes[1], input_shapes[2]),
            name='input_tcr'
        )
        input_covar = tf.keras.layers.Input(
            shape=(input_shapes[3]),
            name='input_covar'
        )
        x = input_covar
        # Linear layers.
        for i in range(n_layers):
            x = tf.keras.layers.Dense(
                units=self.labels_dim,
                activation="relu" if i < n_layers-1 else out_activation,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None
            )(x)
        output = x

        self.training_model = tf.keras.models.Model(
            inputs=[input_tcr, input_covar],
            outputs=output,
            name='model_noseq'
        )

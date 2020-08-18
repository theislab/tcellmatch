import numpy as np
import tensorflow as tf
from typing import Union

from tcellmatch.models.layers.layer_aa_embedding import LayerAaEmbedding
from tcellmatch.models.layers.layer_inception import LayerInception


class ModelInception:

    def __init__(
            self,
            labels_dim: int,
            input_shapes: tuple,
            n_filters_1x1: Union[np.ndarray, list],
            n_filters_out: Union[np.ndarray, list],
            n_hidden: int,
            split: bool,
            final_pool: str = "average",
            residual_connection=False,
            aa_embedding_dim: Union[int, None] = 0,
            depth_final_dense: int = 1,
            out_activation: str = "linear",
            dropout: float = 0.0
    ):
        """ Inception-based feed-forward network.

        Build the feed forward network as a tf.keras.Model object.

        :param n_filters_1x1:
        :param n_filters_out:
        :param n_filters_final:
        :param n_hidden:
        :param split:
        :param final_pool:
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
        :param dropout: drop out rate for lstm.
        """
        self.args = {
            "labels_dim": labels_dim,
            "input_shapes": input_shapes,
            "n_filters_out": n_filters_out,
            "n_hidden": n_hidden,
            "dropout": dropout,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "out_activation": out_activation,
            "final_pool": final_pool,
            "residual_connection": residual_connection,
            "split": split
        }
        self.labels_dim = labels_dim
        self.input_shapes = input_shapes
        self.n_filters_1x1 = n_filters_1x1
        self.n_filters_out = n_filters_out
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.aa_embedding_dim = aa_embedding_dim
        self.depth_final_dense = depth_final_dense
        self.out_activation = out_activation
        self.final_pool = final_pool
        if residual_connection:
            print("WARNING: residual connection in inception model not yet supported.")
        self.residual_connection = residual_connection
        self.split = split
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
        if self.split:
            pep = x[:, self.x_len:, :]
            x = x[:, :self.x_len, :]  # TCR sequence from here on.
        # Inception layers.
        for i, (n0, n1) in enumerate(zip(self.n_filters_1x1, self.n_filters_out)):
            x = LayerInception(
                n_filters_1x1=n0,
                n_filters_out=n1,
                residual_connection=self.residual_connection,
                dropout=self.dropout
            )(x)
            if self.split:
                pep = LayerInception(
                    n_filters_1x1=n0,
                    n_filters_out=n1,
                    residual_connection=self.residual_connection,
                    dropout=self.dropout
                )(pep)
        # Final pooling across sequence positions:
        if self.final_pool.lower() == "average":
            x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(x)
        elif self.final_pool.lower() == "max":
            x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last')(x)
        else:
            raise ValueError("final_pool %s not recognized" % self.final_pool)
        if self.split:
            if self.final_pool.lower() == "average":
                pep = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(pep)
            elif self.final_pool.lower() == "max":
                pep = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last')(pep)
            else:
                raise ValueError("final_pool %s not recognized" % self.final_pool)
            x = tf.concat([x, pep], axis=1)
        # Optional concatenation of non-sequence covariates.
        if input_covar.shape[1] > 0:
            x = tf.concat([x, input_covar], axis=1)
        # Final dense layers.
        for i in range(self.depth_final_dense):
            x = tf.keras.layers.Dense(
                units=self.n_hidden if i < self.depth_final_dense - 1 else self.labels_dim,
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
            name='model_inception'
        )

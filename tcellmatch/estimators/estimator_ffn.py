import numpy as np
import pandas as pd
import pickle
import scipy.sparse
import tensorflow as tf
from typing import Union, List
import os

from tcellmatch.models.models_ffn import ModelBiRnn, ModelSa, ModelConv, ModelLinear, ModelNoseq
from tcellmatch.models.model_inception import ModelInception
from tcellmatch.estimators.additional_metrics import pr_global, pr_label, auc_global, auc_label, \
    deviation_global, deviation_label
from tcellmatch.estimators.estimator_base import EstimatorBase
from tcellmatch.estimators.losses import WeightedBinaryCrossentropy
from tcellmatch.estimators.metrics import custom_r2, custom_logr2


class EstimatorFfn(EstimatorBase):

    model: tf.keras.Model
    model_hyperparam: dict
    train_hyperparam: dict
    history: dict
    evaluations: dict
    evaluations_custom: dict

    def __init__(
            self,
            model_name=None
    ):
        EstimatorBase.__init__(self=self)
        self.model_name = model_name
        self.model_hyperparam = None
        self.train_hyperparam = None
        self.wbce_weight = None

        # Training and evaluation output containers.
        self.history = None
        self.results_test = None
        self.predictions = None
        self.evaluations = None
        self.evaluations_custom = None

    def _out_activation(self, loss) -> str:
        """ Decide whether network output activation

        This decision is based on the loss function.

        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :return: How network output transformed:

            - "categorical_crossentropy", "cce": softmax
            - "binary_crossentropy", "bce": sigmoid
            - "weighted_binary_crossentropy", "wbce": sigmoid
            - "mean_squared_error", "mse": linear
            - "mean_squared_logarithmic_error", "msle": exp
            - "poisson", "pois": exp
        """
        if loss.lower() in ["categorical_crossentropy", "cce"]:
            return "softmax"
        elif loss.lower() in ["binary_crossentropy", "bce"]:
            return "sigmoid"
        elif loss.lower() in ["weighted_binary_crossentropy", "wbce"]:
            return "linear"  # Cost function expect logits.
        elif loss.lower() in ["mean_squared_error", "mse"]:
            return "linear"
        elif loss.lower() in ["mean_squared_logarithmic_error", "msle"]:
            return "exponential"
        elif loss.lower() in ["poisson", "pois"]:
            return "exponential"
        else:
            raise ValueError("Loss %s not recognized." % loss)

    def set_wbce_weight(self, weight):
        """ Overwrites automatically computed weight that is chosen based on training data.

        :param weight: Weight to use.
        :return:
        """
        self.wbce_weight = weight

    def build_bilstm(
            self,
            topology: List[int],
            split: bool = False,
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            residual_connection: bool = False,
            dropout: float = 0.0,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            optimize_for_gpu: bool = True,
            dtype: str = "float32"
    ):
        """ Build a BiLSTM-based feed-forward model to use in the estimator.

        :param topology: The depth of each bilstm layer (length of feature vector)
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param optimize_for_gpu: Whether to choose implementation optimized for GPU.
        :param dtype:
        :return:
        """
        self._build_sequential(
            model="bilstm",
            topology=topology,
            split=split,
            aa_embedding_dim=aa_embedding_dim,
            depth_final_dense=depth_final_dense,
            residual_connection=residual_connection,
            dropout=dropout,
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing,
            optimize_for_gpu=optimize_for_gpu,
            dtype=dtype
        )

    def build_bigru(
            self,
            topology: List[int],
            split: bool = False,
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            residual_connection: bool = False,
            dropout: float = 0.0,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            optimize_for_gpu: bool = True,
            dtype: str = "float32"
    ):
        """ Build a BiGRU-based feed-forward model to use in the estimator.

        :param topology: The depth of each bilstm layer (length of feature vector)
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.s
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param optimize_for_gpu: Whether to choose implementation optimized for GPU.
        :param dtype:
        :return:
        """
        self._build_sequential(
            model="bigru",
            topology=topology,
            split=split,
            aa_embedding_dim=aa_embedding_dim,
            depth_final_dense=depth_final_dense,
            residual_connection=residual_connection,
            dropout=dropout,
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing,
            optimize_for_gpu=optimize_for_gpu,
            dtype=dtype
        )

    def _build_sequential(
            self,
            model: str,
            topology: List[int],
            split: bool,
            aa_embedding_dim: Union[None, int],
            depth_final_dense: int,
            residual_connection: bool,
            dropout: float,
            optimizer: str,
            lr: float,
            loss: str,
            label_smoothing: float,
            optimize_for_gpu: bool,
            dtype: str = "float32"
    ):
        """ Build a BiLSTM-based feed-forward model to use in the estimator.

        :param topology: The depth of each bilstm layer (length of feature vector)
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param optimize_for_gpu: Whether to choose implementation optimized for GPU.
        :param dtype:
        :return:
        """
        # Save model settings:
        self.model_hyperparam = {
            "model": model,
            "topology": topology,
            "split": split,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "residual_connection": residual_connection,
            "dropout": dropout,
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "label_smoothing": label_smoothing,
            "optimize_for_gpu": optimize_for_gpu,
            "dtype": dtype
        }
        self.model = ModelBiRnn(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            model=model.lower(),
            labels_dim=self.y_train.shape[1],
            topology=topology,
            split=split,
            residual_connection=residual_connection,
            aa_embedding_dim=aa_embedding_dim,
            depth_final_dense=depth_final_dense,
            out_activation=self._out_activation(loss=loss),
            dropout=dropout
        )

        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )

    def build_self_attention(
            self,
            attention_size: List[int],
            attention_heads: List[int],
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            residual_connection: bool = False,
            dropout: float = 0.0,
            split: bool = False,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            dtype: str = "float32"
    ):
        """ Build a self-attention-based feed-forward model to use in the estimator.

        :param attention_size: hidden size for attention, could be divided by attention_heads.
        :param attention_heads: number of heads in attention.
        :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """
        # Save model settings:
        self.model_hyperparam = {
            "model": "selfattention",
            "attention_size": attention_size,
            "attention_heads": attention_heads,
            "split": split,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "residual_connection": residual_connection,
            "dropout": dropout,
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "label_smoothing": label_smoothing,
            "dtype": dtype
        }

        # Build model.
        self.model = ModelSa(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            labels_dim=self.y_train.shape[1],
            attention_size=attention_size,
            attention_heads=attention_heads,
            residual_connection=residual_connection,
            split=split,
            aa_embedding_dim=aa_embedding_dim,
            out_activation=self._out_activation(loss=loss),
            depth_final_dense=depth_final_dense,
            dropout=dropout
        )

        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )

    def build_conv(
            self,
            activations: List[str],
            filter_widths: List[int],
            filters: List[int],
            strides: Union[List[Union[int, None]], None] = None,
            pool_sizes: Union[List[Union[int, None]], None] = None,
            pool_strides: Union[List[Union[int, None]], None] = None,
            batch_norm: bool = False,
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            dropout: float = 0.0,
            split: bool = False,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            dtype: str = "float32"
    ):
        """ Build a self-attention-based feed-forward model to use in the estimator.

        :param activations: Activation function. Refer to documentation of tf.keras.layers.Conv2D
        :param filter_widths: Number of neurons per filter. Refer to documentation of tf.keras.layers.Conv2D
        :param filters: NUmber of filters / output channels. Refer to documentation of tf.keras.layers.Conv2D
        :param strides: Stride size for convolution on sequence. Refer to documentation of tf.keras.layers.Conv2D
        :param pool_sizes: Size of max-pooling, ie. number of output nodes to pool over.
            Refer to documentation of tf.keras.layers.MaxPool2D:pool_size
        :param pool_strides: Stride of max-pooling.
            Refer to documentation of tf.keras.layers.MaxPool2D:strides
        :param batch_norm: Whether to perform batch normalization.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """
        # Save model settings.
        self.model_hyperparam = {
            "model": "conv",
            "activations": activations,
            "filter_widths": filter_widths,
            "filters": filters,
            "strides": strides,
            "pool_sizes": pool_sizes,
            "pool_strides": pool_strides,
            "batch_norm": batch_norm,
            "split": split,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "dropout": dropout,
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "label_smoothing": label_smoothing,
            "dtype": dtype
        }

        # Build model.
        self.model = ModelConv(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            labels_dim=self.y_train.shape[1],
            activations=activations,
            filter_widths=filter_widths,
            filters=filters,
            strides=strides,
            pool_sizes=pool_sizes,
            pool_strides=pool_strides,
            batch_norm=batch_norm,
            split=split,
            aa_embedding_dim=aa_embedding_dim,
            out_activation=self._out_activation(loss=loss),
            depth_final_dense=depth_final_dense,
            dropout=dropout
        )
        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )

    def build_inception(
            self,
            n_filters_1x1: List[int],
            n_filters_out: List[int],
            n_hidden: int = 10,
            residual_connection: bool = True,
            aa_embedding_dim: Union[None, int] = None,
            depth_final_dense: int = 1,
            final_pool: str = "average",
            dropout: float = 0.0,
            split: bool = False,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            dtype: str = "float32"
    ):
        """ Build a self-attention-based feed-forward model to use in the estimator.

        :param n_filters_1x1:
        :param n_filters_out:
        :param n_filters_final:
        :param n_hidden:
         :param residual_connection: apply residual connection or not.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param depth_final_dense: Number of final densely connected layers. They all have labels_dim number of units
            and relu activation functions, apart from the last, which has either linear or sigmoid activation,
            depending on out_probabilities.
        :param final_pool:
        :param dropout: Drop-out rate for training.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """
        # Save model settings.
        self.model_hyperparam = {
            "model": "inception",
            "n_filters_1x1": n_filters_1x1,
            "n_filters_out": n_filters_out,
            "n_hidden": n_hidden,
            "split": split,
            "final_pool": final_pool,
            "residual_connection": residual_connection,
            "aa_embedding_dim": aa_embedding_dim,
            "depth_final_dense": depth_final_dense,
            "dropout": dropout,
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "dtype": dtype
        }

        # Build model.
        self.model = ModelInception(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            labels_dim=self.y_train.shape[1],
            n_filters_1x1=n_filters_1x1,
            n_filters_out=n_filters_out,
            n_hidden=n_hidden,
            split=split,
            final_pool=final_pool,
            residual_connection=residual_connection,
            aa_embedding_dim=aa_embedding_dim,
            depth_final_dense=depth_final_dense,
            out_activation=self._out_activation(loss=loss),
            dropout=dropout
        )
        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )

    def build_linear(
            self,
            aa_embedding_dim: Union[None, int] = None,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            dtype: str = "float32"
    ):
        """ Build a linear feed-forward model to use in the estimator.

        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """
        # Save model settings.
        self.model_hyperparam = {
            "model": "linear",
            "aa_embedding_dim": aa_embedding_dim,
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "label_smoothing": label_smoothing,
            "dtype": dtype
        }

        # Build model.
        self.model = ModelLinear(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            labels_dim=self.y_train.shape[1],
            aa_embedding_dim=aa_embedding_dim,
            out_activation=self._out_activation(loss=loss)
        )
        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )

    def build_noseq(
            self,
            optimizer: str = "adam",
            lr: float = 0.005,
            loss: str = "bce",
            label_smoothing: float = 0,
            dtype: str = "float32"
    ):
        """ Build a dense feed-forward model to use in the estimator that does not include the sequence data.

        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param dtype:
        :return:
        """
        # Save model settings.
        self.model_hyperparam = {
            "model": "noseq",
            "optimizer": optimizer,
            "lr": lr,
            "loss": loss,
            "label_smoothing": label_smoothing,
            "dtype": dtype
        }

        # Build model.
        self.model = ModelNoseq(
            input_shapes=(
                self.x_train.shape[1],
                self.x_train.shape[2],
                self.x_train.shape[3],
                self.covariates_train.shape[1],
                self.tcr_len
            ),
            labels_dim=self.y_train.shape[1],
            out_activation=self._out_activation(loss=loss)
        )
        self._compile_model(
            optimizer=optimizer,
            lr=lr,
            loss=loss,
            label_smoothing=label_smoothing
        )

    def _compile_model(
            self,
            optimizer,
            lr,
            loss,
            label_smoothing: float = 0
    ):
        """ Shared model building code across model classes.

        :param optimizer: str optimizer name or instance of tf.keras.optimizers
        :param loss: loss name

            - "categorical_crossentropy", "cce" for multiple boolean binding events with categorical crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
            - "poisson", "pois" for count value prediction based on Poisson log-likelihood.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :return:
        """
        # Instantiate loss.
        if loss.lower() in ["categorical_crossentropy", "cce"]:
            tf_loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False,
                label_smoothing=label_smoothing
            )
            metric_class = "categorical_crossentropy"
        elif loss.lower() in ["binary_crossentropy", "bce"]:
            tf_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=False,
                label_smoothing=label_smoothing
            )
            metric_class = "binary_crossentropy"
        elif loss.lower() in ["weighted_binary_crossentropy", "wbce"]:
            tf_loss = WeightedBinaryCrossentropy(
                weight_positives=1./self.frac_positives - 1. if self.wbce_weight is None else self.wbce_weight,
                label_smoothing=label_smoothing
            )
            metric_class = "binary_crossentropy"
        elif loss.lower() in ["mean_squared_error", "mse"]:
            tf_loss = tf.keras.losses.MeanSquaredError()
            metric_class = "real"
        elif loss.lower() in ["mean_squared_logarithmic_error", "msle"]:
            tf_loss = tf.keras.losses.MeanSquaredLogarithmicError()
            metric_class = "real"
        elif loss.lower() in ["poisson", "pois"]:
            tf_loss = tf.keras.losses.Poisson()  # only in tf>=1.14.1
            metric_class = "real"
        else:
            raise ValueError("Loss %s not recognized." % loss)

        # Assemble metrics.
        if metric_class == "categorical_crossentropy":
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name="keras_acc"),
                tf.keras.metrics.Precision(name="keras_precision"),
                tf.keras.metrics.Recall(name="keras_recall"),
                tf.keras.metrics.AUC(name="keras_auc"),
                tf.keras.metrics.FalseNegatives(name="keras_fn"),
                tf.keras.metrics.FalsePositives(name="keras_fp"),
                tf.keras.metrics.TrueNegatives(name="keras_tn"),
                tf.keras.metrics.TruePositives(name="keras_tp"),
                tf.keras.metrics.CategoricalCrossentropy(name="keras_ce", from_logits=False, label_smoothing=0)
            ]
        elif metric_class == "binary_crossentropy":
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name="keras_acc"),
                tf.keras.metrics.Precision(name="keras_precision"),
                tf.keras.metrics.Recall(name="keras_recall"),
                tf.keras.metrics.AUC(name="keras_auc"),
                tf.keras.metrics.FalseNegatives(name="keras_fn"),
                tf.keras.metrics.FalsePositives(name="keras_fp"),
                tf.keras.metrics.TrueNegatives(name="keras_tn"),
                tf.keras.metrics.TruePositives(name="keras_tp"),
                tf.keras.metrics.BinaryCrossentropy(name="keras_ce", from_logits=False, label_smoothing=0)
            ]
        elif metric_class == "real":
            metrics = [
                tf.keras.metrics.MeanSquaredError(name="keras_mse"),
                tf.keras.metrics.RootMeanSquaredError(name="keras_rmse"),
                tf.keras.metrics.MeanSquaredLogarithmicError(name="keras_msle"),
                tf.keras.metrics.Poisson(name="keras_poisson"),
                tf.keras.metrics.CosineSimilarity(name="keras_cosine"),
                custom_r2,
                custom_logr2
            ]
        else:
            assert False

        # Build optimizer:
        if optimizer.lower() == "adam":
            tf.keras.optimizers.Adam(lr=lr)
        else:
            raise ValueError("optimizer %s not recognized" % optimizer)

        # Compile model.
        self.model.training_model.compile(
            loss=tf_loss,
            optimizer=optimizer,
            metrics=metrics
        )

    def train(
            self,
            epochs: int = 1000,
            batch_size: int = 128,
            max_steps_per_epoch: int = 100,
            validation_split=0.1,
            validation_batch_size: int = 256,
            max_validation_steps: int = 100,
            patience: int = 20,
            lr_schedule_min_lr: float = 1e-5,
            lr_schedule_factor: float = 0.2,
            lr_schedule_patience: int = 5,
            log_dir: Union[str, None] = None,
            use_existing_eval_partition: bool = False
    ):
        """ Train model.

        Uses validation loss and maximum number of epochs as termination criteria.

        :param epochs: refer to tf.keras.models.Model.fit() documentation
        :param steps_per_epoch: refer to tf.keras.models.Model.fit() documentation
        :param batch_size: refer to tf.keras.models.Model.fit() documentation
        :param validation_split: refer to tf.keras.models.Model.fit() documentation
        :param validation_batch_size: Number of validation data observations to evaluate evaluation metrics on.
        :param validation_steps: refer to tf.keras.models.Model.fit() documentation
        :param patience: refer to tf.keras.models.Model.fit() documentation
        :param lr_schedule_min_lr: Minimum learning rate for learning rate reduction schedule.
        :param lr_schedule_factor: Factor to reduce learning rate by within learning rate reduction schedule
            when plateu is reached.
        :param lr_schedule_patience: Patience for learning rate reduction in learning rate reduction schedule.
        :param log_dir: Directory to save tensorboard callback to. Disabled if None.
        :param use_existing_eval_partition: Whether to use existing training-evalutation partition of data. The index
            vectors are expected in self.idx_train and self.idx_eval.
        :return:
        """
        # Save training settings to allow model restoring.
        self.train_hyperparam = {
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "validation_batch_size": validation_batch_size,
            "patience": patience,
            "lr_schedule_min_lr": lr_schedule_min_lr,
            "lr_schedule_factor": lr_schedule_factor,
            "lr_schedule_patience": lr_schedule_patience,
            "log_dir": log_dir
        }
        # Set callbacks.
        cbs = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_schedule_factor,
                patience=lr_schedule_patience,
                min_lr=lr_schedule_min_lr
            )
        ]
        if log_dir is not None:
            cbs.append(tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                write_graph=False,
                write_grads=False,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None,
                embeddings_data=None,
                update_freq='epoch'
            ))

        # Split data into training and evaluation.
        if use_existing_eval_partition:
            idx_val = np.array([self.idx_train_val.tolist().index(x)
                                for x in self.idx_train_val if x in self.idx_val])
            idx_train = np.array([self.idx_train_val.tolist().index(x)
                                  for x in self.idx_train_val if x in self.idx_train])
        else:
            # Split training data into training and evaluation.
            # Perform this splitting based on clonotypes.
            clones = np.unique(self.clone_train)
            clones_eval = clones[np.random.choice(
                a=np.arange(0, clones.shape[0]),
                size=round(clones.shape[0] * validation_split),
                replace=False
            )]
            clones_train = np.array([x for x in clones if x not in clones_eval])
            # Collect observations by clone partition:
            idx_val = np.where([x in clones_eval for x in self.clone_train])[0]
            idx_train = np.where([x in clones_train for x in self.clone_train])[0]
            # Save partitions in terms of original indexing.
            self.idx_train = self.idx_train_val[idx_train]
            self.idx_val = self.idx_train_val[idx_val]
            # Assert that split is exclusive and complete:
            assert len(set(clones_eval).intersection(set(clones_train))) == 0, \
                "ERROR: train-test assignment was not exclusive on level of clones"
            assert len(set(idx_val).intersection(set(idx_train))) == 0, \
                "ERROR: train-test assignment was not exclusive on level of cells"
            assert len(clones_eval) + len(clones_train) == len(clones), \
                "ERROR: train-test split was not complete on the level of clones"
            assert len(idx_val) + len(idx_train) == len(self.clone_train), \
                "ERROR: train-test split was not complete on the level of cells"

        print("Number of observations in evaluation data: %i" % len(idx_val))
        print("Number of observations in training data: %i" % len(idx_train))

        # Build Datasets for each training and evaluation data to feed iterators for each to model fitting.
        train_dataset = tf.data.Dataset.from_tensor_slices((
            (self.x_train[idx_train], self.covariates_train[idx_train]),
            self.y_train[idx_train]
            #self.sample_weight_train[idx_train]
        )).shuffle(buffer_size=len(idx_train), reshuffle_each_iteration=True).\
            repeat().batch(batch_size).prefetch(1)

        eval_dataset = tf.data.Dataset.from_tensor_slices((
            (self.x_train[idx_val], self.covariates_train[idx_val]),
            self.y_train[idx_val]
        )).shuffle(buffer_size=len(idx_val), reshuffle_each_iteration=True).\
            repeat().batch(validation_batch_size).prefetch(1)

        steps_per_epoch = min(max(len(idx_train) // batch_size, 1), max_steps_per_epoch)
        validation_steps = min(max(len(idx_val) // validation_batch_size, 1), max_validation_steps)

        # Fit model and save summary of fitting.
        if len(self.x_train.shape) != 4:
           raise ValueError("input shape should be [?,1,pos,feature]")
        self.history = self.model.training_model.fit(
            x=train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=eval_dataset,
            validation_steps=validation_steps,
            callbacks=cbs,
            verbose=2
        ).history

    @property
    def idx_train_in_train_val(self):
        return np.intersect1d(self.idx_train_val, self.idx_train, return_indices=True)[1]

    @property
    def idx_val_in_train_val(self):
        return np.intersect1d(self.idx_train_val, self.idx_val, return_indices=True)[1]

    def evaluate(
            self,
            batch_size: int = 1024
    ):
        """ Evaluate loss on test data.

        :param batch_size: Batch size for evaluation.
        :return:
        """
        results_test = self.evaluate_any(
            x=self.x_test,
            covar=self.covariates_test,
            y=self.y_test,
            batch_size=batch_size
        )
        results_val = self.evaluate_any(
            x=self.x_train[self.idx_val_in_train_val],
            covar=self.covariates_train[self.idx_val_in_train_val],
            y=self.y_train[self.idx_val_in_train_val],
            batch_size=batch_size,
        )
        results_train = self.evaluate_any(
            x=self.x_train[self.idx_train_in_train_val],
            covar=self.covariates_train[self.idx_train_in_train_val],
            y=self.y_train[self.idx_train_in_train_val],
            batch_size=batch_size,
        )
        self.evaluations = {
            "test": results_test,
            "val": results_val,
            "train": results_train
        }

    def evaluate_any(
            self,
            x,
            covar,
            y,
            batch_size: int = 1024,
    ):
        """ Evaluate loss on supplied data.

        :param batch_size: Batch size for evaluation.
        :return: Dictionary of metrics
        """
        results = self.model.training_model.evaluate(
            x=(x, covar),
            y=y,
            batch_size=batch_size,
            verbose=0
        )
        return dict(zip(self.model.training_model.metrics_names, results))

    def evaluate_custom(
            self,
            classification_metrics: bool = True,
            regression_metrics: bool = False,
            transform: str = None
    ):
        """ Obtain custom evaluation metrics for classification task on train, val and test data.
        """
        results_test = self.evaluate_custom_any(
            yhat=self.predict_any(x=self.x_test, covar=self.covariates_test, batch_size=1024),
            yobs=self.y_test,
            nc=self.nc_test,
            labels=np.asarray(self.peptide_seqs_test),
            labels_unique=self.peptide_seqs_unique,
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            transform_flavour=transform
        )
        results_val = self.evaluate_custom_any(
            yhat=self.predict_any(
                x=self.x_train[self.idx_val_in_train_val],
                covar=self.covariates_train[self.idx_val_in_train_val],
                batch_size=1024
            ),
            yobs=self.y_train[self.idx_val_in_train_val],
            nc=self.nc_train[self.idx_val_in_train_val] if self.nc_train is not None else None,
            labels=np.asarray(self.peptide_seqs_train)[self.idx_val_in_train_val] \
                if self.peptide_seqs_train is not None else None,
            labels_unique=self.peptide_seqs_unique,
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            transform_flavour=transform
        )
        results_train = self.evaluate_custom_any(
            yhat=self.predict_any(
                x=self.x_train[self.idx_train_in_train_val],
                covar=self.covariates_train[self.idx_train_in_train_val],
                batch_size=1024
            ),
            yobs=self.y_train[self.idx_train_in_train_val],
            nc=self.nc_train[self.idx_train_in_train_val] if self.nc_train is not None else None,
            labels=np.asarray(self.peptide_seqs_train)[self.idx_train_in_train_val] \
                if self.peptide_seqs_train is not None else None,
            labels_unique=self.peptide_seqs_unique,
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            transform_flavour=transform
        )
        self.evaluations_custom = {
            "test": results_test,
            "val": results_val,
            "train": results_train
        }

    def _evaluate_custom_any(
            self,
            yhat,
            yobs,
            nc,
            classification_metrics: bool,
            regression_metrics: bool,
            labels=None,
            labels_unique=None,
            transform_flavour: str = None
    ):
        """ Obtain custom evaluation metrics for classification task on any data.
        """
        metrics_global = {}
        metrics_local = {}
        if regression_metrics:
            mse_global, msle_global, r2_global, r2log_global = deviation_global(
                y_hat=[yhat], y_obs=[yobs]
            )
            mse_label, msle_label, r2_label, r2log_label = deviation_label(
                y_hat=[yhat], y_obs=[yobs], labels=[labels], labels_unique=labels_unique
            )
            metrics_global.update({
                "mse": mse_global,
                "msle": msle_global,
                "r2": r2_global,
                "r2log": r2log_global
            })
            metrics_local.update({
                "mse": mse_label,
                "msle": msle_label,
                "r2": r2_label,
                "r2log": r2log_label
            })

        if classification_metrics:
            if transform_flavour is not None:
                yhat, yobs = self.transform_predictions_any(
                    yhat=yhat,
                    yobs=yobs,
                    nc=nc,
                    flavour=transform_flavour
                )
            score_auc_global = auc_global(y_hat=[yhat], y_obs=[yobs])
            prec_global, rec_global, tp_global, tn_global, fp_global, fn_global = pr_global(
                y_hat=[yhat], y_obs=[yobs]
            )
            score_auc_label = auc_label(
                y_hat=[yhat], y_obs=[yobs], labels=[labels], labels_unique=labels_unique
            )
            prec_label, rec_label, tp_label, tn_label, fp_label, fn_label = pr_label(
                y_hat=[yhat], y_obs=[yobs], labels=[labels], labels_unique=labels_unique
            )
            metrics_global.update({
                "auc": score_auc_global,
                "prec": prec_global,
                "rec": rec_global,
                "tp": tp_global,
                "tn": tn_global,
                "fp": fp_global,
                "fn": fn_global
            })
            metrics_local.update({
                "auc": score_auc_label,
                "prec": prec_label,
                "rec": rec_label,
                "tp": tp_label,
                "tn": tn_label,
                "fp": fp_label,
                "fn": fn_label
            })

        return {
            "global": metrics_global,
            "local": metrics_local
        }

    def evaluate_custom_any(
            self,
            yhat,
            yobs,
            nc,
            labels=None,
            labels_unique=None,
            classification_metrics: bool = True,
            regression_metrics: bool = False,
            transform_flavour: str = None
    ):
        """
        Obtain custom evaluation metrics for classification task.

        Ignores labels as samples are not structured by labels (ie one sample contains observations on all labels.

        :param yhat:
        :param yobs:
        :param nc:
        :param labels:
        :param transform_flavour:
        :return:
        """
        return self._evaluate_custom_any(
            yhat=yhat,
            yobs=yobs,
            nc=nc,
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            transform_flavour=transform_flavour,
            labels=None,
            labels_unique=None
        )

    def predict(
            self,
            batch_size: int = 128
    ):
        """ Predict labels on test data.

        :param batch_size: Batch size for evaluation.
        :return:
        """
        self.predictions = self.model.training_model.predict(
            x=(self.x_test, self.covariates_test),
            batch_size=batch_size
        )

    def predict_any(
            self,
            x,
            covar,
            batch_size: int = 128
    ):
        """ Predict labels on any data.

        :param batch_size: Batch size for evaluation.
        :return:
        """
        return self.model.training_model.predict(
            x=(x, covar),
            batch_size=batch_size,
            verbose=0
        )

    def transform_predictions_any(
            self,
            yhat,
            yobs,
            nc,
            flavour="10x_cd8_v1"
    ):
        """ Transform model predictions and ground truth labels on test data.

        Transform predictions and self.y_test

            - "10x_cd8" Use this setting to transform the real valued output of a network trained with MSE loss
                into probability space by using the bound/unbound classifier published with the 10x data set:
                An antigen is bound if it has (1) at least 10 counts and (2) at least 5 times more counts
                than the highest observed negative control and (3) is the highest count pMHC.
                Requires negative controls to be defined during reading.

        :param flavour: Type of transform to use, see function description.
        :return:
        """
        if flavour == "10x_cd8_v1":
            if self.model_hyperparam["loss"] not in ["mse", "msle", "poisson"]:
                raise ValueError("Do not use transform_predictions with flavour=='10x_cd8_v1' on a model fit "
                                 "with a loss that is not mse, msle or poisson.")

            if nc.shape[1] == 0:
                raise ValueError("Negative controls were not set, supply these during data reading.")

            predictions_new = np.zeros(yhat.shape)
            idx_bound_predictions = [np.where(np.logical_and(
                np.logical_and(x > 10., np.max(x) == x),  # At least 10 UMIs and is maximum element of cell.
                x > 5. * np.max(nc[i, :])
                # At least 5x as many UMIs as highest negative control UMI count in cell.
            ))[0] for i, x in enumerate(yhat)]
            for i, j in enumerate(idx_bound_predictions):
                if len(j) > 0:
                    predictions_new[i, j[-1]] = 1.  # Chose last label if two labels are called.
            yhat = predictions_new

            y_test_new = np.zeros(yobs.shape)
            idx_bound_y = [np.where(np.logical_and(
                np.logical_and(x > 10., np.max(x) == x),  # At least 10 UMIs and is maximum element of cell.
                x > 5. * np.max(nc[i, :])
                # At least 5x as many UMIs as highest negative control UMI count in cell.
            ))[0] for i, x in enumerate(yobs)]
            for i, j in enumerate(idx_bound_y):
                if len(j) > 0:
                    y_test_new[i, j[-1]] = 1.  # Chose last label if two labels are called.
            yobs = y_test_new
        else:
            raise ValueError("flavour %s not recognized" % flavour)
        return yhat, yobs

    def transform_predictions(
            self,
            flavour="10x_cd8_v1"
    ):
        """ Transform model predictions and ground truth labels on test data.

        Transform predictions and self.y_test

            - "10x_cd8" Use this setting to transform the real valued output of a network trained with MSE loss
                into probability space by using the bound/unbound classifier published with the 10x data set:
                An antigen is bound if it has (1) at least 10 counts and (2) at least 5 times more counts
                than the highest observed negative control and (3) is the highest count pMHC.
                Requires negative controls to be defined during reading.

        :param flavour: Type of transform to use, see function description.
        :return:
        """
        if flavour == "10x_cd8_v1":
            if self.model_hyperparam["loss"] not in ["mse", "msle", "poisson"]:
                raise ValueError("Do not use transform_predictions with flavour=='10x_cd8_v1' on a model fit "
                                 "with a loss that is not mse, msle or poisson.")

            if self.nc_test.shape[1] == 0:
                raise ValueError("Negative controls were not set, supply these during data reading.")

            predictions_new = np.zeros(self.predictions.shape)
            idx_bound_predictions = [np.where(np.logical_and(
                np.logical_and(x > 10., np.max(x) == x),  # At least 10 UMIs and is maximum element of cell.
                x > 5. * np.max(self.nc_test[i, :])  # At least 5x as many UMIs as highest negative control UMI count in cell.
            ))[0] for i, x in enumerate(self.predictions)]
            for i, j in enumerate(idx_bound_predictions):
                if len(j) > 0:
                    predictions_new[i, j[-1]] = 1.  # Chose last label if two labels are called.
            self.predictions = predictions_new

            y_test_new = np.zeros(self.y_test.shape)
            idx_bound_y = [np.where(np.logical_and(
                np.logical_and(x > 10., np.max(x) == x),  # At least 10 UMIs and is maximum element of cell.
                x > 5. * np.max(self.nc_test[i, :])  # At least 5x as many UMIs as highest negative control UMI count in cell.
            ))[0] for i, x in enumerate(self.y_test)]
            for i, j in enumerate(idx_bound_y):
                if len(j) > 0:
                    y_test_new[i, j[-1]] = 1.  # Chose last label if two labels are called.
            self.y_test = y_test_new
        else:
            raise ValueError("flavour %s not recognized" % flavour)

    def save_results(
            self,
            fn
    ):
        """ Save training history, test loss and test predictions.

        Will generate the following files:

            - fn+"history.pkl": training history dictionary
            - fn+"evaluations.npy": loss on test data
            - fn+"evaluations_custom.npy": loss on test data

        :param self:
        :param fn: Path and file name prefix to write to.
        :param save_labels: Whether to save ground truth labels. Use this for saving disk space.
        :return:
        """
        with open(fn + "_history.pkl", 'wb') as f:
            pickle.dump(self.history, f)
        with open(fn + "_evaluations.pkl", 'wb') as f:
            pickle.dump(self.evaluations, f)
        with open(fn + "_evaluations_custom.pkl", 'wb') as f:
            pickle.dump(self.evaluations_custom, f)
        if self.label_ids is not None:
            pd.DataFrame({"label": self.label_ids}).to_csv(fn + "_labels.csv")
        with open(fn + "_peptide_seqs_unique.pkl", 'wb') as f:
            pickle.dump(self.peptide_seqs_unique, f)

    def load_results(
            self,
            fn
    ):
        """ Load training history, test loss and test predictions.

        Will add the following entries to this instance from files:

            - fn+"history.pkl": training history dictionary
            - fn+"evaluations.npy": loss on test data
            - fn+"evaluations_custom.npy": loss on test data

        :param self:
        :param fn: Path and file name prefix to read from.
        :return:
        """
        with open(fn + "_history.pkl", 'rb') as f:
            self.history = pickle.load(f)
        with open(fn + "_evaluations.pkl", 'rb') as f:
            self.evaluations = pickle.load(f)
        with open(fn + "_evaluations_custom.pkl", 'rb') as f:
            self.evaluations_custom = pickle.load(f)

    def save_model_full(
            self,
            fn,
            reduce_size: bool = False,
            save_yhat: bool = True,
            save_train_data: bool = False
    ):
        """ Save model settings, data and weights.

        Saves all data necessary to perform full one-step model reloading with self.load_model().

        :param self:
        :param fn: Path and file name prefix to write to.
        :param reduce_size: Whether to save storage efficient, ie only elements that are absolutely necessary.
        :return:
        """
        self.save_model(fn=fn)
        self.save_estimator_args(fn=fn)
        self.save_data(
            fn=fn,
            train=save_train_data,
            test=True,
            reduce_size=reduce_size
        )
        if save_yhat:
            self.save_predictions(
                fn=fn,
                train=save_train_data,
                test=True
            )

    def save_model(
            self,
            fn
    ):
        """ Save model weights.

        :param self:
        :param fn: Path and file name prefix to write to. Will be suffixed with .tf to use tf weight saving.
        :return:
        """
        self.model.training_model.save_weights(fn, save_format="tf")

    def load_model_full(
            self,
            fn: str = None,
            fn_settings: str = None,
            fn_data: str = None,
            fn_model: str = None
    ):
        """ Load entire model, this is possible if model weights, data and settings were stored.

        :param self:
        :param fn: Path and file name prefix to read model settings, data and model from.
        :param fn_settings: Path and file name prefix to read model settings from.
        :param fn_data: Path and file name prefix to read all fitting relevant data objects from.
        :param fn_model: Path and file name prefix to read model weights from.
        :param log_dir: Directory to save tensorboard callback to. Disabled if None. This is given to allow the user
             to choose between a new logging directory and the directory from the saved settings.

                - None if you want to enforce no logging.
                - "previous" if you want to use the directory saved in the settings.
                - any other string: This will be the new directory.
        :return:
        """
        if fn is not None:
            fn_settings = fn
            fn_data = fn
            fn_model = fn
        # Initialise model.
        self.load_data(fn=fn_data)
        self.load_model(
            fn_settings=fn_settings,
            fn_model=fn_model
        )

    def load_model(
            self,
            fn_settings: str,
            fn_model: str
    ):
        """ Load model from .tf weights.

        :param self:
        :param fn: Path and file name prefix to read model settings from.
        :return:
        """
        # Initialise model.
        self.load_model_settings(fn=fn_settings)
        self.initialise_model_from_settings()
        self.model.training_model.load_weights(fn_model)

    def save_estimator_args(
            self,
            fn
    ):
        """ Save model settings.

        :param self:
        :param fn: Path and file name prefix to write to.
        :return:
        """
        # Save model args.
        with open(fn + "_model_args.pkl", 'wb') as f:
            pickle.dump(self.model.args, f)
        # Save model settings.
        with open(fn + "_model_settings.pkl", 'wb') as f:
            pickle.dump(self.model_hyperparam, f)
        # Save training settings.
        with open(fn + "_train_settings.pkl", 'wb') as f:
            pickle.dump(self.train_hyperparam, f)

    def load_model_settings(
            self,
            fn
    ):
        """ Load model settings.

        :param self:
        :param fn: Path and file name prefix to read weights from.
        :return:
        """
        # Load model settings.
        with open(fn + "_model_settings.pkl", 'rb') as f:
            self.model_hyperparam = pickle.load(f)
        # Load training settings.
        with open(fn + "_train_settings.pkl", 'rb') as f:
            self.train_hyperparam = pickle.load(f)

    def initialise_model_from_settings(self):
        """

        :return:
        """
        # Build model.
        if self.model_hyperparam["model"].lower() in ["bilstm", "bigru"]:
            self._build_sequential(
                split=self.model_hyperparam["split"],
                model=self.model_hyperparam["model"],
                topology=self.model_hyperparam["topology"],
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                depth_final_dense=self.model_hyperparam["depth_final_dense"],
                residual_connection=self.model_hyperparam["residual_connection"],
                dropout=self.model_hyperparam["dropout"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                optimize_for_gpu=self.model_hyperparam["optimize_for_gpu"],
                dtype=self.model_hyperparam["dtype"]
            )
        elif self.model_hyperparam["model"].lower() in ["sa", "selfattention"]:
            self.build_self_attention(
                attention_size=self.model_hyperparam["attention_size"],
                attention_heads=self.model_hyperparam["attention_heads"],
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                depth_final_dense=self.model_hyperparam["depth_final_dense"],
                residual_connection=self.model_hyperparam["residual_connection"],
                dropout=self.model_hyperparam["dropout"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                dtype=self.model_hyperparam["dtype"]
            )
        elif self.model_hyperparam["model"].lower() in ["conv", "convolutional"]:
            self.build_conv(
                activations=self.model_hyperparam["activations"],
                filter_widths=self.model_hyperparam["filter_widths"],
                filters=self.model_hyperparam["filters"],
                strides=self.model_hyperparam["strides"],
                pool_sizes=self.model_hyperparam["pool_sizes"],
                pool_strides=self.model_hyperparam["pool_strides"],
                batch_norm=self.model_hyperparam["batch_norm"],
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                depth_final_dense=self.model_hyperparam["depth_final_dense"],
                dropout=self.model_hyperparam["dropout"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                dtype=self.model_hyperparam["dtype"]
            )
        elif self.model_hyperparam["model"].lower() in ["inception"]:
            self.build_inception(
                split=self.model_hyperparam["split"],
                n_filters_1x1=self.model_hyperparam["n_filters_1x1"],
                n_filters_out=self.model_hyperparam["n_filters_out"],
                n_hidden=self.model_hyperparam["n_hidden"],
                final_pool=self.model_hyperparam["final_pool"],
                residual_connection=self.model_hyperparam["residual_connection"],
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                depth_final_dense=self.model_hyperparam["depth_final_dense"],
                dropout=self.model_hyperparam["dropout"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                dtype=self.model_hyperparam["dtype"]
            )
        elif self.model_hyperparam["model"].lower() in ["linear"]:
            self.build_linear(
                aa_embedding_dim=self.model_hyperparam["aa_embedding_dim"],
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                dtype=self.model_hyperparam["dtype"]
            )
        elif self.model_hyperparam["model"].lower() in ["noseq"]:
            self.build_noseq(
                optimizer=self.model_hyperparam["optimizer"],
                lr=self.model_hyperparam["lr"],
                loss=self.model_hyperparam["loss"],
                label_smoothing=self.model_hyperparam["label_smoothing"],
                dtype=self.model_hyperparam["dtype"]
            )
        else:
            assert False

    def save_weights_tonumpy(
            self,
            fn
    ):
        """ Save model weights to pickled list of numpy arrays.

        :param fn: Path and file name prefix to write to.
        :return:
        """
        weights = self.model.training_model.get_weights()
        with open(fn + "_weights.pkl", 'wb') as f:
            pickle.dump(weights, f)

    def load_weights_asnumpy(
            self,
            fn
    ):
        """ Load model weights.

        :param fn: Path and file name prefix to write to.
        :return: List of model weights as numpy arrays.
        """
        with open(fn + "_weights.pkl", 'rb') as f:
            weights = pickle.load(f)
        return weights

    def save_data(
            self,
            fn,
            train: bool,
            test: bool,
            reduce_size: bool = False
    ):
        """ Save train and test data.

        :param fn: Path and file name prefix to write all fitting relevant data objects to.
        :param reduce_size: Whether to save storage efficient, ie only elements that are absolutely necessary.
        :return:
        """
        if train:
            if not reduce_size:
                scipy.sparse.save_npz(
                    matrix=scipy.sparse.csr_matrix(np.reshape(self.x_train, [self.x_train.shape[0], -1])),
                    file=fn + "_x_train.npz"
                )
            np.save(arr=self.x_train.shape, file=fn + "_x_train_shape.npy")
            if not reduce_size and self.covariates_train.shape[1] > 0:
                if not isinstance(self.covariates_train, scipy.sparse.csr_matrix):
                    covariates_train = scipy.sparse.csr_matrix(np.reshape(
                        self.covariates_train,
                        [self.covariates_train.shape[0], -1]
                    ))
                else:
                    covariates_train = self.covariates_train
                scipy.sparse.save_npz(matrix=covariates_train, file=fn + "_covariates_train.npz")
            np.save(arr=self.covariates_train.shape, file=fn + "_covariates_train_shape.npy")
            if not reduce_size:
                if not isinstance(self.y_train, scipy.sparse.csr_matrix):
                    y_train = scipy.sparse.csr_matrix(self.y_train)
                else:
                    y_train = self.y_train
                scipy.sparse.save_npz(matrix=y_train, file=fn + "_y_train.npz")
            np.save(arr=self.y_train.shape, file=fn + "_y_train_shape.npy")
            if not reduce_size and self.nc_train is not None and self.nc_train.shape[1] > 0:
                if not isinstance(self.nc_train, scipy.sparse.csr_matrix):
                    nc_train = scipy.sparse.csr_matrix(self.nc_train)
                else:
                    nc_train = self.nc_train
                scipy.sparse.save_npz(matrix=nc_train, file=fn + "_nc_train.npz")
            if self.nc_train is not None:
                np.save(arr=self.nc_train.shape, file=fn + "_nc_train_shape.npy")
            else:
                np.save(arr=np.array([None]), file=fn + "_nc_train_shape.npy")
            np.save(arr=self.clone_train, file=fn + "_clone_train.npy")
            if self.peptide_seqs_train is not None:
                pd.DataFrame({"antigen": self.peptide_seqs_train}).to_csv(fn + "_peptide_seqs_train.csv")

        if self.x_test is not None and test:
            if not reduce_size:
                scipy.sparse.save_npz(
                    matrix=scipy.sparse.csr_matrix(np.reshape(self.x_test, [self.x_test.shape[0], -1])),
                    file=fn + "_x_test.npz"
                )
            np.save(arr=self.x_test.shape, file=fn + "_x_test_shape.npy")
            if not reduce_size and self.covariates_test.shape[1] > 0:
                if not isinstance(self.covariates_test, scipy.sparse.csr_matrix):
                    covariates_test = scipy.sparse.csr_matrix(np.reshape(
                        self.covariates_test,
                        [self.covariates_test.shape[0], -1]
                    ))
                else:
                    covariates_test = self.covariates_test
                scipy.sparse.save_npz(matrix=covariates_test, file=fn + "_covariates_test.npz")
            np.save(arr=self.covariates_test.shape, file=fn + "_covariates_test_shape.npy")
            if not reduce_size:
                if not isinstance(self.y_test, scipy.sparse.csr_matrix):
                    y_test = scipy.sparse.csr_matrix(self.y_test)
                else:
                    y_test = self.y_test
                scipy.sparse.save_npz(matrix=y_test, file=fn + "_y_test.npz")
            np.save(arr=self.y_test.shape, file=fn + "_y_test_shape.npy")
            if not reduce_size and self.nc_test is not None and self.nc_test.shape[1] > 0:
                if not isinstance(self.nc_test, scipy.sparse.csr_matrix):
                    nc_test = scipy.sparse.csr_matrix(self.nc_test)
                else:
                    nc_test = self.nc_test
                scipy.sparse.save_npz(matrix=nc_test, file=fn + "_nc_test.npz")
            if self.nc_test is not None:
                np.save(arr=self.nc_test.shape, file=fn + "_nc_test_shape.npy")
            else:
                np.save(arr=np.array([None]), file=fn + "_nc_test_shape.npy")
            np.save(arr=self.clone_test, file=fn + "_clone_test.npy")
            if self.peptide_seqs_test is not None:
                pd.DataFrame({"antigen": self.peptide_seqs_test}).to_csv(fn + "_peptide_seqs_test.csv")

        pd.DataFrame({"antigen": self.peptide_seqs_unique}).to_csv(fn + "_peptide_seqs_unique.csv")
        self.save_idx(fn=fn)

    def load_data(
            self,
            fn
    ):
        """ Load train and test data.

        Note: Cryptic numpy pickle error is thrown if a csr_matrix containing only a single None is loaded.

        :param fn: Path and file name prefix to read all fitting relevant data objects from.
        :return:
        """
        x_train_shape = np.load(file=fn + "_x_train_shape.npy")
        if os.path.isfile(fn + "_x_train.npz"):
            self.x_train = np.reshape(np.asarray(
                scipy.sparse.load_npz(file=fn + "_x_train.npz").todense()
            ), x_train_shape)
        else:
            # Fill x with small all zero array to allow model loading.
            self.x_train = np.zeros(x_train_shape)
        covariates_train_shape = np.load(file=fn + "_covariates_train_shape.npy")
        if os.path.isfile(fn + "_covariates_train.npz") and covariates_train_shape[1] > 0:
            self.covariates_train = np.reshape(np.asarray(scipy.sparse.load_npz(
                file=fn + "_covariates_train.npz"
            ).todense()), covariates_train_shape)
        else:
            self.covariates_train = np.zeros(covariates_train_shape)
        self.x_len = x_train_shape[2]
        if os.path.isfile(fn + "_y_train_shape.npy"):
            y_train_shape = np.load(file=fn + "_y_train_shape.npy")
        else:
            y_train_shape = None
        if os.path.isfile(fn + "_y_train.npz"):
            self.y_train = np.asarray(scipy.sparse.load_npz(file=fn + "_y_train.npz").todense())
        else:
            if y_train_shape is not None:  # depreceated, remove
                self.y_train = np.zeros(y_train_shape)
        if os.path.isfile(fn + "_nc_train.npz"):
            self.nc_train = np.asarray(scipy.sparse.load_npz(file=fn + "_nc_train.npz").todense())
        else:
            self.nc_train = None
        self.clone_train = np.load(file=fn + "_clone_train.npy")

        if os.path.isfile(fn + "_x_test_shape.npy"):
            x_test_shape = np.load(file=fn + "_x_test_shape.npy")
            if os.path.isfile(fn + "_x_test.npz"):
                self.x_test = np.reshape(np.asarray(
                    scipy.sparse.load_npz(file=fn + "_x_test.npz").todense()
                ), x_test_shape)
            covariates_test_shape = np.load(file=fn + "_covariates_test_shape.npy")
            if os.path.isfile(fn + "_covariates_test.npz") and covariates_test_shape[1] > 0:
                self.covariates_test = np.reshape(np.asarray(scipy.sparse.load_npz(
                    file=fn + "_covariates_test.npz"
                ).todense()), covariates_test_shape)
            else:
                self.covariates_test = np.zeros(covariates_test_shape)
            if os.path.isfile(fn + "_y_test_shape.npy"):
                y_test_shape = np.load(file=fn + "_y_test_shape.npy")
            else:
                y_test_shape = None
            if os.path.isfile(fn + "_y_test.npz"):
                self.y_test = np.asarray(scipy.sparse.load_npz(file=fn + "_y_test.npz").todense())
            else:
                if y_test_shape is not None:  # depreceated, remove
                    self.y_test = np.zeros(y_test_shape)
            if os.path.isfile(fn + "_nc_test.npz"):
                self.nc_test = np.asarray(scipy.sparse.load_npz(file=fn + "_nc_test.npz").todense())
            else:
                self.nc_test = None
            self.clone_test = np.load(file=fn + "_clone_test.npy")

        self.load_idx(fn=fn)

    def save_predictions(
            self,
            fn,
            train: bool,
            test: bool
    ):
        """ Save predictions.

        :param fn: Path and file name prefix to write all fitting relevant data objects to.
        :param reduce_size: Whether to save storage efficient, ie only elements that are absolutely necessary.
        :return:
        """
        if train:
            yhat_train = self.predict_any(x=self.x_train, covar=self.covariates_train)
            np.save(arr=yhat_train, file=fn + "_yhat_train.npy")

        if self.x_test is not None and test:
            yhat_test = self.predict_any(x=self.x_test, covar=self.covariates_test)
            np.save(arr=yhat_test, file=fn + "_yhat_test.npy")

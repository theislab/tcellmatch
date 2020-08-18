from typing import Union

from tcellmatch.estimators.estimator_ffn import EstimatorFfn


class EstimatorBinary(EstimatorFfn):

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
        Obtain custom evaluation metrics.

        Overwrites parent method to stratify samples by label (antigen) that is predicted.

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
            labels=labels,
            labels_unique=labels_unique
        )

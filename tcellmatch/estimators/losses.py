import tensorflow as tf


class WeightedBinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):

    def __init__(
            self,
            weight_positives: float = 1,
            label_smoothing: float = 0
    ):
        """ Build instance of weighted binary crossentropy based on tf.keras.losses.BinaryCrossentropy implementation.

        :param weight_positives: Factor to multiply binary crossentropy cost of positive observation labels with.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        """
        super(WeightedBinaryCrossentropy, self).__init__(
            from_logits=True,
            label_smoothing=label_smoothing
        )
        self.weight_positives = weight_positives

    def call(self, y_true, y_pred):
        """ Computes weighted binary crossentropy loss for a batch.

        :param y_true: Observations (observations, labels).
        :param y_pred: Predictions (observations, labels).
        :return: Loss (observations, labels).
        """
        # Format data.
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # Perform label smoothing.
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute loss.
        loss = tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true,
            logits=y_pred,
            pos_weight=self.weight_positives
        )
        return loss

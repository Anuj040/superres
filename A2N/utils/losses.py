"""module for loss and metric definitions"""
import tensorflow as tf
import tensorflow.keras.backend as K


class PSNRLayer(tf.keras.metrics.Metric):
    """Custom keras metric layer for Peak Signal-to-Noise Ratio between two tensors"""

    def __init__(self, max_value: float = 2.0, name: str = "PSNR", **kwargs):
        """
        Args:
            max_value (float, optional): tensor's range. Default = 2.0 for image in (-1.0, 1.0)
            name (str, optional): Layer name. Defaults to "PSNR".
        """
        super().__init__(name=name, **kwargs)
        # Stores the summation of metric value over the whole dataset
        self.metric = self.add_weight(name="PSNR", initializer="zeros")
        # Samples count
        self.metric_count = self.add_weight(name="Count", initializer="zeros")
        # Function for calculating the metric value
        self.m = PSNR(max_value=max_value)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Metric values for a given batch
        metric_value = self.m(y_true, y_pred)
        # Number of samples in a given batch
        count = tf.cast(K.shape(metric_value)[0], self.dtype)
        # Sum of metric value for the processed samples
        self.metric.assign_add(K.sum(metric_value))
        # Total number of samples processed
        self.metric_count.assign_add(count)

    def result(self):
        # Average metric value
        return self.metric / self.metric_count

    def reset_states(self):
        # metric state reset at the start of each epoch.
        self.metric.assign(0.0)
        self.metric_count.assign(0)


def PSNR(max_value: float = 2.0) -> tf.Tensor:
    """method for calculating Peak Signal-to-Noise Ratio between two tensors

    Args:
        max_value (float, optional): tensor's range. Defaults to 2.0 for image scaled to (-1.0, 1.0)
    """

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        diff = y_true - y_pred
        rmse = K.sqrt(K.mean(diff * diff, axis=[-3, -2, -1]))
        del y_true, y_pred
        return 20.0 * (K.log(max_value / rmse)) / K.log(10.0)

    _loss.__name__ = "PSNR"

    return _loss

"""module for loss and metric definitions"""
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


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

    def update_state(self, y_true, y_pred, *args):
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


def perceptual_layer(weights: float = 1.0, drop: float = 0.5) -> KM.Model:
    """
    Preceptual loss on latent features as in Johnson et al 2015 https://arxiv.org/abs/1603.08155

    Args:
        weights (float, optional): weighing for the loss. Defaults to 1.0.
        drop (float, optional): randomly drop channels for regularization. Defaults to 0.5.

    Returns:
        KM.Model: perceptual error calculating model
    """
    # pylint: disable = E1123, E1124, E1120

    # Ground truth
    input1 = KL.Input(shape=(None, None, 3))
    # Model output
    input2 = KL.Input(shape=(None, None, 3))
    # concat the two tensors above
    input_tensor = tf.concat([input1, input2], axis=0)

    # VGG19 feature extractor trained on Imagenet
    feature_extractor = tf.keras.applications.VGG19(
        include_top=False, weights="imagenet"
    )

    # input_tensor is normalized between (-1.0, 1.0)
    # extractor compatible format
    inputs = 127.5 * (input_tensor + 1.0)
    inputs = tf.keras.applications.vgg19.preprocess_input(inputs, data_format=None)
    feature_maps = feature_extractor(inputs)

    # Extract deep features for the GT and generated image
    content, generated = tf.split(feature_maps, num_or_size_splits=2, axis=0)

    # Randomly zero-out some feature differences
    drop_features = KL.SpatialDropout2D(rate=drop)(content - generated)
    error = (weights / (1 - drop)) * K.mean(K.square(drop_features))
    return KM.Model(inputs=[input1, input2], outputs=error, name="vgg")


if __name__ == "__main__":
    loss_fn = perceptual_layer()
    a = tf.random.uniform(shape=(2, 256, 256, 3), dtype=tf.float32)
    b = tf.random.uniform(shape=(2, 256, 256, 3), dtype=tf.float32)
    loss = loss_fn([a, b])
    print(loss)

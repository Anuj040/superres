"""custom model trainer"""
from typing import Dict

import tensorflow as tf
import tensorflow.keras.models as KM

from A2N.model_utils.gan import Discriminator
from A2N.utils.losses import perceptual_layer


class Trainer(KM.Model):  # pylint: disable=too-many-ancestors
    """custom model trainer function

    Args:
        KM.Model: parent class
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals

    def __init__(self, model: KM.Model, mode: str = "norm") -> None:
        """
        Args:
            model (KM.Model): super-resolution model
            mode (str, optional): training mode. One of ["norm", "gan"]
        """
        super().__init__()
        self.model = model
        assert mode in ["norm", "gan"], "only 'norm' or 'gan' train mode"

        if mode == "gan":
            # Discriminator model for GAN training
            self.discriminator = Discriminator()
        self.mode = mode

    def compile(
        self,
        optimizer: tf.keras.optimizers,
        loss: Dict[str, tf.keras.losses.Loss],
        loss_weights: Dict[str, float],
        metric=Dict[str, tf.keras.metrics.Metric],
        perceptual: bool = False,
    ) -> None:
        # pylint: disable=attribute-defined-outside-init
        """compiles the model object with corresponding attributes
        Args:
            optimizer (tf.keras.optimizers): optimizer for model training
            loss (Dict[str, tf.keras.losses.Loss]): loss definitions for the model outputs
            loss_weights (Dict[str, float]): weights for each loss definition
            metric (Dict[str, tf.keras.metrics.Metric]): performance metrics for model outputs
            perceptual (bool, optional): use precpetual loss on latent feature. Defaults to False
        """
        super().compile()
        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = loss_weights

        if perceptual:
            loss["percep"] = perceptual_layer()
            metric["percep"] = None
        assert len(self.loss) == len(
            metric
        ), "provide metric functions for all outputs, 'None' wherever not applicable"
        self.loss_metrics = metric
        self.loss_keys = loss.keys()

        if self.mode == "gan":

            # optimizer and loss fn. for gan training
            self.d_optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-5, beta_1=0.0, beta_2=0.9  # Default values from A2N
            )
            self.d_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """method to process model call
        Args:
            inputs (tf.Tensor): input tensor
        Returns:
            tf.Tensor: model output
        """
        return self.model(inputs, training=False)

    def train_step(self, inputs: tf.Tensor) -> dict:
        """method to implement a training step
        Args:
            inputs (tf.Tensor): a batch instance consisting of input-output pair
        Returns:
            dict: dict object containing batch performance parameters
        """

        # unpack the data
        LR, HR = inputs

        # build the graph
        with tf.GradientTape() as tape:

            # get the model outputs
            model_outputs = self.model(LR, training=True)

            losses = []
            losses_grad = []
            # calculate losses
            for i, key in enumerate(self.loss_keys):
                if key == "percep":
                    losses.append(self.loss[key]([HR, model_outputs], training=False))
                else:
                    losses.append(self.loss[key](HR, model_outputs))
                losses_grad.append(tf.reduce_mean(losses[i]) * self.loss_weights[key])

            if self.mode == "gan":
                d_gen = self.discriminator(model_outputs)

                # Disc score loss for generator update
                g_loss = self.d_loss(d_gen, tf.ones_like(d_gen))
                losses_grad.append(g_loss)

        # calculate and apply gradients
        grads = tape.gradient(
            losses_grad,
            self.model.trainable_weights,
        )
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # prepare the logs dictionary
        logs = dict(zip(self.loss_keys, losses[:-1] if self.mode == "gan" else losses))
        logs = {key: tf.reduce_mean(value) for key, value in logs.items()}

        # Add metrics if applicable
        for _, key in enumerate(self.loss_keys):
            metric_func = self.loss_metrics[key]

            # Only evaluate the not-None metrics
            if metric_func is not None:
                metric_func.update_state(HR, model_outputs)
                logs[metric_func.name] = metric_func.result()

        # House-keeping
        del tape

        if self.mode == "gan":
            # Discriminator training
            # build the graph
            with tf.GradientTape() as tape:
                d_gen = self.discriminator(model_outputs)
                d_HR = self.discriminator(HR)

                # Disc score loss on generated and gt images
                d_loss = self.d_loss(d_gen, tf.zeros_like(d_gen)) + self.d_loss(
                    d_HR, tf.ones_like(d_HR)
                )

            # calculate and apply gradients
            grads = tape.gradient(
                [d_loss],
                self.discriminator.trainable_weights,
            )
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
            logs["d_loss"] = d_loss
            # House - keeping
            del tape

        return logs

    @property
    def metrics(self):
        # list `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # Without this property, `reset_states()` will have to called
        # manually.
        return [
            metric_func
            for key, metric_func in self.loss_metrics.items()
            if metric_func is not None
        ]

    def test_step(self, inputs: tf.Tensor) -> dict:
        """method to implement evaluation step
        Args:
            inputs(tf.Tensor): a batch instance consisting of input-output pair
        Returns:
            dict: dict object containing batch performance parameters on classification task only
        """
        # unpack the data
        LR, HR = inputs

        # get the model outputs
        model_outputs = self.model(LR, training=False)

        # Get the metric calculator
        metric_func = self.loss_metrics["HR"]

        # Calculate the performance metrics
        metric_func.update_state(HR, model_outputs)

        # return the logs dict
        return {metric_func.name: metric_func.result()}

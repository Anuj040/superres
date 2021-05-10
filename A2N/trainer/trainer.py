"""custom model trainer"""
import tensorflow as tf
import tensorflow.keras.models as KM


class Trainer(KM.Model):  # pylint: disable=too-many-ancestors
    """custom model trainer function

    Args:
        KM.Model: parent class
    """

    def __init__(self, model: KM.Model) -> None:
        """
        Args:
            model (KM.Model): super-resolution model
        """
        super().__init__()
        self.model = model

    def compile(
        self, optimizer: tf.keras.optimizers, loss: tf.keras.losses.Loss
    ) -> None:
        # pylint: disable=attribute-defined-outside-init
        """compiles the model object with corresponding attributes
        Args:
            optimizer (tf.keras.optimizers): optimizer for model training
            loss (Dict[tf.keras.losses.Loss]): loss definitions for the model outputs
        """
        super().compile()
        self.optimizer = optimizer
        self.loss = loss
        self.loss_keys = loss.keys()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """method to process model call
        Args:
            inputs (tf.Tensor): input tensor
        Returns:
            tf.Tensor: model output
        """
        return self.model(inputs, training=False)

    def train_step(self, inputs) -> dict:
        """method to implement a training step
        Args:
            data (tf.Tensor): a batch instance consisting of input-output pair
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
            # calculate losses
            for i, key in enumerate(self.loss_keys):
                losses.append(self.loss[key](HR, model_outputs))

        # calculate and apply gradients
        grads = tape.gradient(
            losses,
            self.model.trainable_weights,
        )
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # prepare the logs dictionary
        logs = dict(zip(self.loss_keys, losses))
        logs = {key: tf.reduce_mean(value) for key, value in logs.items()}

        # # Add metrics if applicable
        # for i, key in enumerate(self.loss_keys):
        #     metric_func = self.loss_metrics[key]

        #     # Only evaluate the not-None metrics
        #     if metric_func is not None:
        #         metric_func.update_state(outputs[i], model_outputs[i])
        #         logs[metric_func.name] = metric_func.result()

        # House-keeping
        del tape
        return logs

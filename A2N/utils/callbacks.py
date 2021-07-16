"""module with custom callbacks"""

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class SaveModel(Callback):

    """custom model saving callback. Normal ModelCheckpoint callback when used with custom trainer,
        saves the whole trainer object, from which I have not been able to retrieve the model object
        as of now. This callback is to go around that issue.

    Args:
        Callback: Parent Class
    """

    def __init__(
        self,
        metric: str = "val_PSNR",
        mode: str = "max",
        thresh: float = 0.0,
        path: str = "save_model",
    ) -> None:
        """initialize the model save callback

        Args:
            metric (str, optional): metric to be tracked. Defaults to "val_PSNR".
            mode (str, optional): desired trend for the metric. Defaults to "max".
            thresh (float, optional): thresholding the metric value. Defaults to 0.0.
            path (str, optional): model save directory. Defaults to "save_model".
        """

        super().__init__()
        self.metric = metric
        self.mode = mode
        self.thresh = K.variable(thresh)
        self.path = path

    def on_epoch_end(self, epoch: int, logs: dict):
        """executes the callback on epoch end

        Args:
            epoch (int): epoch number
            logs (dict): performance logs
        """
        # Metric calculated once per "n" epochs
        metric = logs.get(self.metric, 0.0)

        if self.mode == "max":
            if metric > self.thresh:
                self.model.model.save(
                    f"{self.path}/model_{epoch+1:04d}_{metric:.4f}.h5"
                )
                K.set_value(self.thresh, metric)

        elif self.mode == "min":
            if metric < self.thresh:
                self.model.model.save(
                    f"{self.path}/model_{epoch+1:04d}_{metric:.4f}.h5"
                )
                K.set_value(self.thresh, metric)
        else:
            raise NotImplementedError(
                f"{self.mode} mode has not been implemented for this callback"
            )

        return super().on_epoch_end(epoch, logs=logs)


def scheduler(epoch: int, lr: float, gamma: float = 0.5, step: int = 200) -> float:
    """learning rate scheduler

    Args:
        epoch (int): current epoch number
        lr (float): [description]
        gamma (float, optional): decay factor. Defaults to 0.5.
        step (int, optional): lr decays after this many epochs. Defaults to 200.

    Returns:
        float: decayed lr
    """
    if not (epoch + 1) % step:
        lr = lr * gamma

    return lr


class LR_Scheduler(Callback):
    """custom learning rate scheduler callback"""

    def __init__(self, gamma: float = 0.5, step: int = 200):
        """
        Args:
            gamma (float, optional): decay factor. Defaults to 0.5.
            step (int, optional): lr decays after this many epochs. Defaults to 200.
        """
        super().__init__()
        self.gamma = gamma
        self.step = step

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        """update the learning rate at the start of the epoch"""

        # Update lr for discriminator
        lr = K.get_value(self.model.d_optimizer.lr)
        lr = scheduler(epoch, lr, self.gamma, self.step)
        K.set_value(self.model.d_optimizer.lr, lr)

        # Update lr for generator
        lr = K.get_value(self.model.optimizer.lr)
        lr = scheduler(epoch, lr, self.gamma, self.step)
        K.set_value(self.model.optimizer.lr, lr)

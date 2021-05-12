"""
    Main module to define, compile and train the NN model
    Paper: https://arxiv.org/abs/2104.09497
    Code Reference: https://github.com/haoyuc/A2N
"""

import os
import sys
from typing import List

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint

sys.path.append("./")
from A2N.trainer.trainer import Trainer
from A2N.utils.generator import DataGenerator
from A2N.utils.losses import PSNRLayer


def AAM(
    input_tensor: tf.Tensor,
    features: int = 40,
    reduction: int = 4,
    n_branch: int = 2,
    temp: float = 30.0,
) -> tf.Tensor:
    """Attention Dropout Module

    Args:
        input_tensor (tf.Tensor):
        features (int, optional): number of features for input tensor. Defaults to 40.
        reduction (int, optional): feature redcution ratio. Defaults to 4.
        n_branch (int, optional): Number of branches to be weighted. Defaults to 2.
        temp (float, optional): temperature for logit softening. Defaults to 30.0.

    Returns:
        tf.Tensor: attention weights for respective branches
    """

    # pylint: disable=unnecessary-lambda
    x = KL.Lambda(lambda x: tf.reduce_mean(x, axis=(1, 2), keepdims=True))(input_tensor)
    x = KL.Dense(features // reduction, use_bias=False)(x)
    x = KL.Activation("relu")(x)
    x = KL.Dense(n_branch, use_bias=False)(x)
    ax = KL.Softmax()(x / temp)

    return ax


def attention_branch(
    input_tensor: tf.Tensor, features: int = 40, kernel: int = 3
) -> tf.Tensor:
    """attention weighted feaures module

    Args:
        input_tensor (tf.Tensor):
        features (int, optional): number of features for the conv layer. Defaults to 40.
        kernel (int, optional): kernel size for the conv layer. Defaults to 3.

    Returns:
        tf.Tensor: attention weighted feautures
    """

    ##### Attention
    # 3x3 convolution
    ax = KL.Conv2D(features, kernel, padding="same", use_bias=False)(input_tensor)
    ax = KL.LeakyReLU(alpha=0.2)(ax)

    # 1x1 convolution features->features
    ax = KL.Conv2D(features, 1)(ax)
    ax = KL.Activation("sigmoid")(ax)

    # Attention weighted features
    # 3x3 convolution
    x = KL.Conv2D(features, kernel, padding="same", use_bias=False)(input_tensor)
    out = x * ax

    # Final 3x3 convolution
    out = KL.Conv2D(features, kernel, padding="same", use_bias=False)(out)
    return out


def AAB(shape: tuple, features: int = 40, name: str = "AAB") -> KM.Model:
    """Attention on Attention block (AAB) model

    Args:
        shape (tuple): shape for the input tensor
        features (int, optional): Number of feautures for the conv/dense layers. Defaults to 40.
        name (str, optional): Object name. Defaults to "AAB".

    Returns:
        KM.Model: AAB model object
    """
    input_tensor = KL.Input(shape=shape, name=name + "_input")
    residual = input_tensor

    x = KL.Conv2D(features, kernel_size=1, use_bias=False)(input_tensor)
    x = KL.LeakyReLU(alpha=0.2)(x)

    # Attention Dropout Module
    ax = AAM(x, features=features)

    # attention weighted features
    attention = attention_branch(x, features=features, kernel=3)

    # features without attention weighting
    non_attention = KL.Conv2D(features, 3, padding="same", use_bias=False)(x)

    # linear combination of features from respective branches above
    x = attention * tf.expand_dims(
        ax[..., 0], axis=-1
    ) + non_attention * tf.expand_dims(ax[..., 1], axis=-1)
    x = KL.LeakyReLU(alpha=0.2)(x)

    out = KL.Conv2D(features, kernel_size=1, use_bias=False)(x)

    # residual connection
    out += residual
    return KM.Model(inputs=input_tensor, outputs=out, name=name)


def pixel_attention(input_tensor: tf.Tensor, features: int = 24) -> tf.Tensor:
    """pixel attention module. Weighs each pixel location with corresponding relevance"""

    x = KL.Conv2D(features, 1)(input_tensor)
    y = KL.Activation("sigmoid")(x)

    return x * y


class SuperRes:
    """super-resolution model class"""

    def __init__(self, scale: int = 4) -> None:
        """
        Args:
            scale (int, optional): upscale factor for the input image. Defaults to 4.
        """
        self.scale = scale
        self.model = self.build(name="superres")

    def build(
        self,
        shape: tuple = (None, None, 3),
        features: int = 40,
        up_features: int = 24,
        n_blocks: int = 16,
        name: str = "superres",
    ) -> KM.Model:
        """super-resolution model builder

        Args:
            shape (tuple, optional): input image size. Defaults to (64, 64, 3).
            features (int, optional): features in attention truck. Defaults to 40.
            up_features (int, optional): features for upsampling blocks. Defaults to 24.
            n_blocks (int, optional): number of AAB blocks. Defaults to 16.
            name (str, optional): name for model object. Defaults to "superres".

        Returns:
            KM.Model: super resoultion model object
        """
        input_tensor = KL.Input(shape=shape, name="LR")

        # First Convolution
        x = KL.Conv2D(features, 3, strides=(1, 1), use_bias=True, padding="same")(
            input_tensor
        )
        fea = x

        # repeated attention-on-attention blocks
        for i in range(n_blocks):
            x = AAB(shape=(shape[0], shape[1], features), name=f"AAB_{i+1}")(x)

        x = KL.Conv2D(features, 3, strides=(1, 1), use_bias=True, padding="same")(x)
        fea = fea + x

        # upscale the features
        if self.scale == 4:
            # 2X
            fea = KL.UpSampling2D(size=(2, 2), interpolation="nearest", name="map2up")(
                fea
            )

            fea = KL.Conv2D(up_features, 3, 1, use_bias=True, padding="same")(fea)
            fea = pixel_attention(fea, features=up_features)
            fea = KL.LeakyReLU(alpha=0.2)(fea)

            fea = KL.Conv2D(up_features, 3, 1, use_bias=True, padding="same")(fea)
            fea = KL.LeakyReLU(alpha=0.2)(fea)

            # 4X
            fea = KL.UpSampling2D(size=(2, 2), interpolation="nearest", name="map4up")(
                fea
            )

            fea = KL.Conv2D(up_features, 3, 1, use_bias=True, padding="same")(fea)
            fea = pixel_attention(fea, features=up_features)
            fea = KL.LeakyReLU(alpha=0.2)(fea)

            fea = KL.Conv2D(up_features, 3, 1, use_bias=True, padding="same")(fea)
            fea = KL.LeakyReLU(alpha=0.2)(fea)

        # Final convolution to get the desired #channels
        # Features for HR reconstruction
        output = KL.Conv2D(shape[-1], 3, 1, use_bias=True, padding="same")(fea)

        # Upscale the input tensor
        up_LR = KL.UpSampling2D(
            size=(self.scale, self.scale), interpolation="bilinear", name="up_LR"
        )(input_tensor)

        # HR reconstruction
        # pylint: disable=unnecessary-lambda
        output = KL.Lambda(lambda x: tf.identity(x), name="HR")(output + up_LR)

        return KM.Model(inputs=input_tensor, outputs=output, name=name)

    def callbacks(self) -> List[Callback]:
        """method to compile all callbacks in one place

        Returns:
            List[Callback]: list of callbacks
        """

        def scheduler(
            epoch: int, lr: float, gamma: float = 0.5, step: int = 200
        ) -> float:
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

        # Step decay callback
        StepLR = LearningRateScheduler(schedule=scheduler, verbose=0)

        # callback for saving model
        save_model = ModelCheckpoint(
            "save_model/model_{epoch:04d}_{val_PSNR:.4f}.h5",
            monitor="val_PSNR",
            save_best_only=True,
            mode="max",
        )

        return [StepLR, save_model]

    def train(
        self,
        train_batch_size: int = 2,
        val_batch_size: int = 2,
        lr: float = 5e-4,
        epochs: int = 10,
        perceptual: bool = False,
    ) -> None:
        """model training method

        Args:
            train_batch_size (int, optional): batch size for training epoch. Defaults to 2.
            val_batch_size (int, optional): batch size for validation epoch. Defaults to 2.
            lr (float, optional): learning rate for optimizer. Defaults to 5e-4.
            epochs (int, optional): number of training epochs. Defaults to 10.
            perceptual (bool, optional): use precpetual loss on latent features. Defaults to False
        """

        # Get the generator objects
        train_generator = DataGenerator(
            "datasets",
            "train",
            batch_size=train_batch_size,
            scale=self.scale,
            shuffle=True,
        )
        val_generator = DataGenerator("datasets", "val", batch_size=val_batch_size)

        # Prepare the trainer object
        model = Trainer(model=self.model)

        # Attributes for the trainer object
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = {"HR": tf.keras.losses.mae}
        metric = {"HR": PSNRLayer()}

        # Compile the trainer object
        model.compile(
            optimizer=optimizer, loss=loss, metric=metric, perceptual=perceptual
        )

        # Number of validation steps
        val_size = len(val_generator)

        # Directory for storing the model file
        os.makedirs("save_model", exist_ok=True)

        # model training
        model.fit(
            train_generator(),
            epochs=epochs,
            workers=8,
            verbose=1,
            validation_data=val_generator(),
            validation_steps=val_size,
            validation_freq=1,
            callbacks=self.callbacks(),
        )


if __name__ == "__main__":
    superres = SuperRes()
    superres.train()

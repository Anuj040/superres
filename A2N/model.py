"""Main module to define, compile and train the NN model"""
import sys

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

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
        shape: tuple = (64, 64, 3),
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

        # tensor resizer object
        resizer = tf.compat.v1.image.resize

        # upscale the features
        if self.scale == 4:
            # 2X
            fea = resizer(
                fea, size=(int(shape[0] * 2), int(shape[1] * 2)), method="nearest"
            )

            fea = KL.Conv2D(up_features, 3, 1, use_bias=True, padding="same")(fea)
            fea = pixel_attention(fea, features=up_features)
            fea = KL.LeakyReLU(alpha=0.2)(fea)

            fea = KL.Conv2D(up_features, 3, 1, use_bias=True, padding="same")(fea)
            fea = KL.LeakyReLU(alpha=0.2)(fea)

            # 4X
            fea = resizer(
                fea, size=(int(shape[0] * 4), int(shape[1] * 4)), method="nearest"
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
        up_LR = resizer(
            input_tensor,
            size=(int(shape[0] * self.scale), int(shape[1] * self.scale)),
            method="bilinear",
            align_corners=False,
        )

        # HR reconstruction
        # pylint: disable=unnecessary-lambda
        output = KL.Lambda(lambda x: tf.identity(x), name="HR")(output + up_LR)

        return KM.Model(inputs=input_tensor, outputs=output, name=name)

    def train(self) -> None:
        """model training method"""

        # Get the generator object
        train_generator = DataGenerator(
            "datasets", "train", scale=self.scale, shuffle=True
        )

        # Prepare the trainer object
        model = Trainer(model=self.model)

        # Attributes for the trainer object
        optimizer = tf.keras.optimizers.Adam()
        loss = {"HR": tf.keras.losses.mse}
        metric = {"HR": PSNRLayer()}

        # Compile the trainer object
        model.compile(optimizer=optimizer, loss=loss, metric=metric)

        # model training
        model.fit(train_generator(), epochs=10, workers=8, verbose=1)


if __name__ == "__main__":
    superres = SuperRes()
    superres.train()

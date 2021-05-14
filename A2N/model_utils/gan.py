"""Module for gan trainer models"""
from typing import Union

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


def basic_block(
    features: int,
    kernel: int,
    stride=1,
    bias=False,
    bn=True,
    act: Union[str, KL.Layer] = KL.Activation("relu"),
):
    """basic block for discriminator model

    Args:
        features (int): number for layers in convolution
        kernel (int): kernel size
        stride (int, optional): stride size. Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Use BatchNorm. Defaults to True.
        act (Union[str, KL.Layer], optional): Activation to use. Defaults to KL.Activation("relu").
    """

    def layer(input_tensor: tf.Tensor):
        x = KL.Conv2D(features, kernel, strides=stride, padding="same", use_bias=bias)(
            input_tensor
        )

        if bn:
            x = KL.BatchNormalization()(x)
        if act is not None:
            x = act(x)
        return x

    return layer


def Discriminator(features: int = 64, depth: int = 7) -> KM.Model:
    """Discriminator model for judging image quality

    Args:
        features (str, int): Number of features for first layer. Defaults to 64.
        depth (str, int): #layers. Defaults to 7

    Returns:
        KM.Model: discriminator model
    """

    input_tensor = KL.Input(shape=(None, None, 3))

    act = KL.LeakyReLU(alpha=0.2)

    x = basic_block(features=features, kernel=3, bn=True, act=act)(input_tensor)

    for i in range(depth):
        if i % 2 == 1:
            stride = 1
            features *= 2
        else:
            stride = 2
        x = basic_block(features, 3, stride=stride, act=act)(x)
    score = KL.Dense(1024)(x)
    score = KL.Dense(1024)(act(score))

    # Reshape into 1D score vectors
    score = KL.Flatten(name="d_score")(score)

    return KM.Model(inputs=input_tensor, outputs=score)


if __name__ == "__main__":
    model = Discriminator()
    model.summary()

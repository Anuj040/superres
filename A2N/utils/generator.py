"""Data generator module"""
from typing import Any, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


class DataGenerator:
    """Datagenerator class"""

    def __init__(
        self,
        path: str,
        split: str = "train",
        batch_size: int = 2,
        shuffle: bool = False,
    ) -> None:
        """Method to build the dataset generator object

        Args:
            path (str): Path to the main directory
            split (str, optional): Data split to utilize. Defaults to "train".
            shuffle (bool, optional): whether to shuffle the data files. Defaults to False.
        """
        # Make single image object retriever function
        builder = tfds.ImageFolder(path)
        dataset = builder.as_dataset(
            split=split, as_supervised=False, shuffle_files=shuffle
        )

        # Make image pairs
        dataset = dataset.map(
            self.pair_maker, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Make batches
        dataset = dataset.batch(batch_size, drop_remainder=split == "train")

        self.dataset = dataset

    def pair_maker(self, element: dict) -> Tuple[tf.Tensor, tf.Tensor]:
        """method to generate a pair of high resolution (HR) and low resolution (LR) images

        Args:
            element (dict): Input object containing image and associated attributes

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Pair of HR-LR images
        """

        # Get the image array # HR image
        image = 2.0 * tf.cast(element["image"], tf.float32) / 255.0 - 1.0

        # shape of original image
        shape = tf.cast(tf.shape(image), tf.float32)

        # dimensions of scaled down image
        lr_h = tf.cast(0.25 * shape[0], tf.int32)
        lr_w = tf.cast(0.25 * shape[1], tf.int32)

        # LR image
        image_small = tf.image.resize(image, size=[lr_h, lr_w])

        return image_small, image

    def __call__(self, *args: Any, **kwds: Any) -> tf.data.Dataset:
        """Method to return the generator object

        Returns:
            tf.data.Dataset: Generator object
        """
        return self.dataset


if __name__ == "__main__":
    train_generator = DataGenerator("datasets", "train", shuffle=True)
    for item in train_generator().take(2):
        print(item)
"""Data generator module"""
from typing import Any

import tensorflow as tf
import tensorflow_datasets as tfds


class DataGenerator:
    """Datagenerator class"""

    def __init__(self, path: str, split: str = "train", shuffle: bool = False) -> None:
        """Method to build the dataset generator object

        Args:
            path (str): Path to the main directory
            split (str, optional): Data split to utilize. Defaults to "train".
            shuffle (bool, optional): whether to shuffle the data files. Defaults to False.
        """
        builder = tfds.ImageFolder(path)
        dataset = builder.as_dataset(
            split=split, as_supervised=False, shuffle_files=shuffle
        )

        self.dataset = dataset

    def __call__(self, *args: Any, **kwds: Any) -> tf.data.Dataset:
        """Method to return the generator object

        Returns:
            tf.data.Dataset: Generator object
        """
        return self.dataset


if __name__ == "__main__":
    train_generator = DataGenerator("datasets", "train", shuffle=True)
    for item in train_generator().take(4):
        print(item["image/filename"])

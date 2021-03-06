"""Data generator module"""
from typing import Any, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def dimension_adjuster(
    *inputs: Tuple[tf.Tensor, tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Method to drop extra dimension coming from ImageDatagenerators

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: scale and dimension adjusted tensors
    """
    Low_res, High_res = inputs
    return (
        2.0 * tf.squeeze(Low_res, axis=1) / 255.0 - 1.0,
        2.0 * tf.squeeze(High_res, axis=1) / 255.0 - 1.0,
    )


def random_apply(func, p: float, image: tf.Tensor) -> tf.Tensor:
    """Randomly apply function func to image with probability p."""

    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32),
        ),
        lambda: func(image),
        lambda: image,
    )


def random_rotate(image: tf.Tensor) -> tf.Tensor:
    """method to randomly rotate the image through [1, 2, 3]*90 degrees

    Args:
        image (tf.Tensor): Image tensor

    Returns:
        tf.Tensor: rotated image tensor
    """
    # get the multiple of 90
    k = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
    return tf.image.rot90(image, k=k)


class DataGenerator:
    """Datagenerator class"""

    def __init__(
        self,
        path: str,
        split: str = "train",
        batch_size: int = 2,
        scale: int = 4,
        shuffle: bool = False,
        augment: bool = False,
    ) -> None:
        """Method to build the dataset generator object

        Args:
            path (str): Path to the main directory
            split (str, optional): Data split to utilize. Defaults to "train".
            batch_size (int, optional): Batch size for the model. Defaults to 2.
            scale (int, optional): Image upsampling ratio. Defaults to 4.
            shuffle (bool, optional): whether to shuffle the data files. Defaults to False.
            augment (bool, optional): Use data augmentation. Defaults to False.
        """
        self.scale = scale
        self.epoch_size = 0
        self.augment = augment

        if split == "train":
            # Make single image object retriever function
            builder = tfds.ImageFolder(path)
            dataset = builder.as_dataset(
                split=split, as_supervised=False, shuffle_files=shuffle
            )
            # Make image pairs
            dataset = dataset.map(
                self.crop_taker, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            # Make batches
            dataset = dataset.batch(batch_size, drop_remainder=split == "train")

            dataset = dataset.map(
                self.pair_maker, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        elif split == "val":
            LR_datagen = ImageDataGenerator()
            HR_datagen = ImageDataGenerator()

            # Provide the same seed for reproducibility
            seed = 1

            # Data generator for LR images
            LR_generator = LR_datagen.flow_from_directory(
                f"{path}/{split}/input",
                class_mode=None,
                seed=seed,
                target_size=(512, 512),
                color_mode="rgb",
                batch_size=1,
                shuffle=shuffle,
            )
            # Data generator for HR images
            HR_generator = HR_datagen.flow_from_directory(
                f"{path}/{split}/gt",
                class_mode=None,
                seed=seed,
                target_size=(2048, 2048),
                color_mode="rgb",
                batch_size=1,
                shuffle=shuffle,
            )
            assert len(HR_generator.filenames) == len(
                LR_generator.filenames
            ), "ensure equal number of HR and LR images"

            # size of validation dataset
            val_size = len(HR_generator.filenames)

            # combine generators into one which yields LR-HR pair
            combined_generator = zip(LR_generator, HR_generator)
            dataset = tf.data.Dataset.from_generator(
                lambda: combined_generator,
                output_signature=(
                    tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32),
                ),
            )
            dataset = dataset.batch(batch_size, drop_remainder=split == "train")

            # Recale the images and drop extra dimension
            dataset = dataset.map(
                dimension_adjuster, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            # ImageDatagenerator gives infinite samples, so to limit that.
            self.epoch_size = (
                val_size // batch_size + 1
                if val_size % batch_size
                else val_size // batch_size
            )

        dataset = dataset.prefetch(20)
        self.dataset = dataset

    def crop_taker(self, element: dict) -> tf.Tensor:
        """method to extract a random crop of high resolution (HR) image

        Args:
            element (dict): Input object containing image and associated attributes

        Returns:
            tf.Tensor: Image crop
        """

        # Get the image crop of given size
        image = tf.image.random_crop(
            element["image"],
            size=(256, 256, 3),
        )
        if self.augment:
            image = random_apply(random_rotate, 0.5, image)
            image = random_apply(tf.image.flip_left_right, 0.5, image)

        # HR image (-1.0, 1.0)
        image = 2.0 * tf.cast(image, tf.float32) / 255.0 - 1.0

        return image

    def pair_maker(self, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """method to generate a pair of high resolution (HR) and low resolution (LR) images

        Args:
            images (tf.Tensor): Batch of Augmented HR image crop

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Pair of HR-LR images
        """

        # shape of original image
        shape = tf.shape(images)

        # dimensions of scaled down image
        lr_h = shape[1] // self.scale
        lr_w = shape[2] // self.scale

        # LR image
        images_small = tf.image.resize(images, size=(lr_h, lr_w), method="bicubic")

        return images_small, images

    def __call__(self, *args: Any, **kwds: Any) -> tf.data.Dataset:
        """Method to return the generator object

        Returns:
            tf.data.Dataset: Generator object
        """
        return self.dataset

    def __len__(self) -> int:
        """Get the total number of batches"""
        epoch_size = self.dataset.cardinality().numpy()
        return epoch_size if epoch_size > 0 else self.epoch_size


if __name__ == "__main__":
    train_generator = DataGenerator("datasets", "train", shuffle=True)
    val_generator = DataGenerator("datasets", "val")
    print(len(train_generator))
    print(len(val_generator))
    for i, item in enumerate(val_generator().take(2)):
        LR, HR = item
        for j, _ in enumerate(LR):
            tf.keras.preprocessing.image.save_img(
                f"datasets/val/LR_{2*i+j}.png",
                LR[j],
                scale=True,
            )
            tf.keras.preprocessing.image.save_img(
                f"datasets/val/HR_{2*i+j}.png",
                HR[j],
                scale=True,
            )

"""main control module"""
import sys

from absl import app, flags

sys.path.append("./")
from A2N.model import SuperRes

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode", "train", ["train", "eval"], "model use mode")
flags.DEFINE_float("lr", 5e-4, "learning rate for the optimizer")
flags.DEFINE_integer("train_batch_size", 2, "batch size for training")
flags.DEFINE_integer("val_batch_size", 2, "validation batch size")
flags.DEFINE_integer("epochs", 10, "training epochs")


def main(argv):
    """main control method"""

    superres = SuperRes()
    if FLAGS.mode == "train":
        superres.train(
            train_batch_size=FLAGS.train_batch_size,
            val_batch_size=FLAGS.val_batch_size,
            lr=FLAGS.lr,
            epochs=FLAGS.epochs,
        )


if __name__ == "__main__":
    app.run(main)

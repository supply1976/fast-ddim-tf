import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def save_images(data: np.ndarray, labels: np.ndarray, output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    num_images = data.shape[0]
    data = data.astype(np.uint8)

    for idx in range(num_images):
        array = data[idx]
        label = int(labels[idx][0] if labels.ndim > 1 else labels[idx])
        output_path = output_dir / f"{prefix}_{idx:05d}_{label}.png"
        Image.fromarray(array).save(output_path)

    print(f"Saved {num_images} images to {output_dir}")


def export_cifar10(root_dir: Path, force: bool = False):
    root_dir = root_dir.expanduser().resolve()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_dir = root_dir / "train"
    test_dir = root_dir / "test"

    if not force and train_dir.exists() and any(train_dir.iterdir()):
        print(f"Train images already exist in {train_dir}. Use --force to regenerate.")
    else:
        save_images(x_train, y_train, train_dir, prefix="train")

    if not force and test_dir.exists() and any(test_dir.iterdir()):
        print(f"Test images already exist in {test_dir}. Use --force to regenerate.")
    else:
        save_images(x_test, y_test, test_dir, prefix="test")

    print(f"CIFAR-10 export complete. Images saved under {root_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Load CIFAR-10 via tf.keras.datasets and save PNG images to datasets/cifar10.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./datasets/cifar10"),
        help="Target output directory for CIFAR-10 PNG images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate CIFAR-10 PNG images even if the destination directory already contains files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_cifar10(args.root, force=args.force)


if __name__ == "__main__":
    main()

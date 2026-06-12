#!/usr/bin/env python3
"""Run CIFAR-10 scheduler/prediction-type diffusion experiments.

This script expands a base config into a 3x2 sweep:

  scheduler in {linear, cosine}
  pred_type in {velocity, noise}

For each split it:
  1. writes a derived training/generation config,
  2. runs training sequentially,
  3. finds the latest EMA epoch checkpoint,
  4. generates 50000 images as NPZ batches,
  5. computes FID against CIFAR-10 test images.

The script intentionally shells out to scripts/run.py so it reuses the same CLI
path as normal training and generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from scipy.linalg import sqrtm


ROOT = Path(__file__).resolve().parent.parent
RUN_PY = ROOT / "scripts" / "run.py"


@dataclass(frozen=True)
class Split:
    scheduler: str
    pred_type: str

    @property
    def name(self) -> str:
        return f"{self.scheduler}_{self.pred_type}"


SPLITS = [
    Split("linear", "velocity"),
    Split("linear", "noise"),
    Split("cosine", "velocity"),
    Split("cosine", "noise"),
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(config, sort_keys=False))


def update_config_for_split(
    base: dict[str, Any],
    split: Split,
    output_root: Path,
    num_gen_images: int,
    gen_batch_size: int | None,
    reverse_steps: int | None,
    disable_inline_gen: bool,
) -> dict[str, Any]:
    cfg = deep_copy_config(base)
    cfg["DIFFUSION_SCHEDULER"]["SCHEDULER"] = split.scheduler
    cfg["DIFFUSION_SCHEDULER"]["PRED_TYPE"] = split.pred_type

    cfg["TRAINING"]["OUTPUT_DIR"] = str(output_root / split.name)
    cfg["TRAINING"]["HYPER_PARAMETERS"]["SAVE_PERIOD"] = 10
    if disable_inline_gen:
        cfg["TRAINING"].setdefault("INLINE_GEN", {})["ENABLE"] = False

    imgen = cfg["IMAGE_GENERATION"]
    imgen["MODEL_PATH"] = ""
    imgen["GEN_TASK"] = "random"
    imgen["NUM_GEN_IMAGES"] = int(num_gen_images)
    imgen["BATCH_SIZE"] = gen_batch_size
    if reverse_steps is not None:
        imgen["REVERSE_STEPS"] = int(reverse_steps)
    imgen["RANDOM_SEED"] = None
    imgen["TARGET_IMAGE_SIZE"] = cfg["NETWORK"]["IMAGE_SIZE"]
    imgen["OUTPUT_OPTIONS"]["SAVE_DIR"] = str(output_root / split.name / "generated")
    imgen["OUTPUT_OPTIONS"]["SAVE_INTERMEDIATE"] = False
    imgen["OUTPUT_OPTIONS"]["SAVE_FORMAT"] = "npz"
    imgen.setdefault("CONDITIONING", {})["EXTERNAL_INPUT"] = None
    imgen["CONDITIONING"]["CLASS_LABEL"] = None
    return cfg


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def find_latest_run_dir(split_output_dir: Path) -> Path:
    train_logs = sorted(
        split_output_dir.glob("**/train.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not train_logs:
        raise FileNotFoundError(f"No train.log found under {split_output_dir}")
    return train_logs[0].parent


def find_latest_ema_epoch_checkpoint(run_dir: Path) -> Path:
    epoch_pattern = re.compile(r"epoch_(\d+)_ema\.keras$")
    candidates: list[tuple[int, float, Path]] = []
    for path in run_dir.glob("*_ema.keras"):
        match = epoch_pattern.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path.stat().st_mtime, path))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    fallback = sorted(run_dir.glob("*_ema.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"No EMA checkpoint found in {run_dir}")


def find_latest_generation_dir(save_root: Path) -> Path:
    imgen_logs = sorted(
        save_root.glob("**/imgen.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not imgen_logs:
        raise FileNotFoundError(f"No imgen.log found under {save_root}")
    return imgen_logs[0].parent


def load_generated_npz_images(generated_root: Path) -> np.ndarray:
    arrays = []
    for path in sorted(generated_root.glob("**/*.npz")):
        with np.load(path) as data:
            if "images" not in data:
                continue
            arrays.append(data["images"].astype(np.float32))
    if not arrays:
        raise FileNotFoundError(f"No generated NPZ files with 'images' key found under {generated_root}")
    images = np.concatenate(arrays, axis=0)
    return np.clip(images, 0.0, 1.0)


def load_real_cifar10_images(dataset_path: Path, count: int) -> np.ndarray:
    test_dir = dataset_path.parent / "test" if dataset_path.name == "train" else dataset_path
    image_paths = sorted(
        list(test_dir.glob("*.png"))
        + list(test_dir.glob("*.jpg"))
        + list(test_dir.glob("*.jpeg"))
    )
    if image_paths:
        images = []
        for path in image_paths[:count]:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            images.append(arr)
        return np.stack(images, axis=0)

    # Fallback is convenient, but may download data if TF has not cached CIFAR-10.
    import tensorflow as tf

    (_, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    return x_test[:count]


def inception_features(
    images: np.ndarray,
    batch_size: int,
    inception_weights: str | None,
) -> np.ndarray:
    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

    weights: str | None = inception_weights if inception_weights else "imagenet"
    model = InceptionV3(include_top=False, pooling="avg", weights=weights, input_shape=(299, 299, 3))

    feats = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        if batch.shape[-1] == 1:
            batch = np.repeat(batch, 3, axis=-1)
        elif batch.shape[-1] == 2:
            batch = np.concatenate([batch, np.zeros_like(batch[..., :1])], axis=-1)
        elif batch.shape[-1] > 3:
            batch = batch[..., :3]
        tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        tensor = tf.image.resize(tensor, (299, 299), method="bilinear")
        tensor = preprocess_input(tensor * 255.0)
        feats.append(model(tensor, training=False).numpy())
    return np.concatenate(feats, axis=0)


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(fid)


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "scheduler",
        "pred_type",
        "fid",
        "num_real_images",
        "num_generated_images",
        "checkpoint",
        "generated_dir",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-10 scheduler/prediction sweep.")
    parser.add_argument("--base-config", type=Path, default=Path("configs/cifar10_32x32.yaml"))
    parser.add_argument("--work-dir", type=Path, default=Path("experiments/cifar10_scheduler_pred_sweep"))
    parser.add_argument("--output-root", type=Path, default=Path("training_outputs/cifar10_scheduler_pred_sweep"))
    parser.add_argument("--num-gen-images", type=int, default=50000)
    parser.add_argument("--gen-batch-size", type=int, default=100)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--reverse-steps", type=int, default=None)
    parser.add_argument("--inception-weights", type=str, default=None, help="Path to local InceptionV3 no-top weights. Defaults to Keras 'imagenet'.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true", help="Reuse existing latest checkpoints.")
    parser.add_argument("--skip-generation", action="store_true", help="Reuse existing generated NPZ files.")
    parser.add_argument(
        "--enable-inline-gen",
        action="store_true",
        help="Keep inline generation enabled during training. Disabled by default for sweep speed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = (ROOT / args.base_config).resolve() if not args.base_config.is_absolute() else args.base_config
    work_dir = (ROOT / args.work_dir).resolve() if not args.work_dir.is_absolute() else args.work_dir
    output_root = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root

    base_cfg = load_yaml(base_config_path)
    config_dir = work_dir / "configs"
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    real_count = args.num_gen_images
    dataset_path = Path(base_cfg["DATASET"]["PATH"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    results: list[dict[str, Any]] = []
    real_features: np.ndarray | None = None
    num_real_images: int | None = None

    for split in SPLITS:
        print(f"\n=== Split: {split.name} ===", flush=True)
        split_cfg = update_config_for_split(
            base_cfg,
            split,
            output_root,
            args.num_gen_images,
            args.gen_batch_size,
            args.reverse_steps,
            not args.enable_inline_gen,
        )
        split_config_path = config_dir / f"{split.name}.yaml"
        write_yaml(split_config_path, split_cfg)

        if not args.skip_training:
            run_command([sys.executable, str(RUN_PY), "--config", str(split_config_path), "--training"], args.dry_run)

        if args.dry_run:
            continue

        split_output_dir = output_root / split.name
        run_dir = find_latest_run_dir(split_output_dir)
        checkpoint = find_latest_ema_epoch_checkpoint(run_dir)
        print(f"Using checkpoint: {checkpoint}", flush=True)

        split_cfg["IMAGE_GENERATION"]["MODEL_PATH"] = str(checkpoint)
        split_config_path = config_dir / f"{split.name}.yaml"
        write_yaml(split_config_path, split_cfg)

        generated_save_root = Path(split_cfg["IMAGE_GENERATION"]["OUTPUT_OPTIONS"]["SAVE_DIR"])
        if not args.skip_generation:
            run_command([sys.executable, str(RUN_PY), "--config", str(split_config_path), "--imgen"], args.dry_run)

        generated_dir = find_latest_generation_dir(generated_save_root)
        generated_images = load_generated_npz_images(generated_dir)[: args.num_gen_images]
        if generated_images.shape[0] < args.num_gen_images:
            raise RuntimeError(
                f"{split.name} generated only {generated_images.shape[0]} images; "
                f"expected {args.num_gen_images}"
            )

        if real_features is None:
            real_images = load_real_cifar10_images(dataset_path, real_count)
            num_real_images = int(real_images.shape[0])
            print(f"Loaded {num_real_images} real images for FID reference", flush=True)
            real_features = inception_features(real_images, args.fid_batch_size, args.inception_weights)

        num_generated_images = int(generated_images.shape[0])
        print(f"Loaded {num_generated_images} generated images for FID[{split.name}]", flush=True)
        fake_features = inception_features(generated_images, args.fid_batch_size, args.inception_weights)
        fid = calculate_fid(real_features, fake_features)
        print(f"FID[{split.name}] = {fid:.6f}", flush=True)

        row = {
            "split": split.name,
            "scheduler": split.scheduler,
            "pred_type": split.pred_type,
            "fid": f"{fid:.6f}",
            "num_real_images": num_real_images,
            "num_generated_images": num_generated_images,
            "checkpoint": str(checkpoint),
            "generated_dir": str(generated_dir),
        }
        results.append(row)
        write_results_csv(results_dir / "fid_results.csv", results)
        with (results_dir / "fid_results.json").open("w") as f:
            json.dump(results, f, indent=2)

    if args.dry_run:
        print(f"\nDry run complete. Configs written to {config_dir}", flush=True)
    else:
        print(f"\nResults written to {results_dir / 'fid_results.csv'}", flush=True)


if __name__ == "__main__":
    main()

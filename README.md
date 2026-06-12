# fast-ddim-tf

Fast TensorFlow implementation of DDIM with configurable U-Net, custom diffusion schedules, and image generation utilities.

## Overview

This repository implements a fast DDIM-style diffusion pipeline in TensorFlow. It supports:

- training on image datasets from `.npz`, `.jpg`, `.png` files
- configurable network architecture and scheduler via YAML configs
- image generation with DDIM reverse sampling
- inline generation during training for monitoring
- CIFAR-10 utility script using `tf.keras.datasets`

## Quickstart

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Generate a config template:

```bash
python scripts/run.py --get-template configs/default.yaml
```

3. Train with a config:

```bash
python scripts/run.py --config configs/cifar10_32x32.yaml --training
```

4. Generate images from a trained model:

```bash
python scripts/run.py --config configs/cifar10_32x32.yaml --imgen
```

## CIFAR-10 scheduler/prediction sweep

Run the 3x2 experiment matrix for:

- `SCHEDULER`: `linear`, `cosine`
- `PRED_TYPE`: `velocity`, `noise`

The automation script trains each split sequentially, generates 50,000 images from
the latest EMA epoch checkpoint, and computes FID against CIFAR-10 test images:

```bash
python scripts/run_cifar10_scheduler_pred_sweep.py
```

Use `--dry-run` to write the split configs and print the commands without
starting training:

```bash
python scripts/run_cifar10_scheduler_pred_sweep.py --dry-run
```

Results are written under `experiments/cifar10_scheduler_pred_sweep/results/`.

## CIFAR-10 dataset helper

Use `utils/download_cifar10.py` to save CIFAR-10 images to `./datasets/cifar10`:

```bash
python utils/download_cifar10.py
```

## Package entrypoint

After install, the command line entrypoint is:

```bash
fast-ddim-tf
```

## Repository layout

- `src/fddim/`: core TensorFlow model and utilities
- `configs/`: YAML examples and templates
- `utils/`: dataset scripts and helpers
- `scripts/`: runnable entrypoints

## Notes

- The implementation uses TensorFlow/Keras.
- The main training and generation logic is in `src/fddim/cli.py`.
- `src/fddim/diffusion_utils.py` contains the diffusion schedule and DDIM reverse sampling math.

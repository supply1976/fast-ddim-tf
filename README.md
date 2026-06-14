# fast-ddim-tf

Fast TensorFlow implementation of DDIM with configurable U-Net, custom diffusion schedules, and image generation utilities.

## Overview

This repository implements a fast DDIM-style diffusion pipeline in TensorFlow. It supports:

- training on image datasets from `.npz`, `.jpg`, `.png` files
- configurable network architecture and scheduler via YAML configs
- image generation with DDIM reverse sampling
- inline generation during training for monitoring
- CIFAR-10 utility script using `tf.keras.datasets`

## Math notes

The implementation follows the derivation in
[`docs/math_note/Diffusion_Model_Mathematics_Notes.pdf`](docs/math_note/Diffusion_Model_Mathematics_Notes.pdf).
The corresponding LaTeX source is available at
[`docs/math_note/Diffusion_Model_Mathematics_Notes.tex`](docs/math_note/Diffusion_Model_Mathematics_Notes.tex).
These notes describe the forward process, DDPM/DDIM reverse updates, and
prediction parameterizations used by `src/fddim/diffusion_utils.py`.
The derivation is written in continuous time and supports arbitrary reverse
time pairs `(s, t)` with `s < t`; the current sampler implementation uses a
uniform reverse timestep grid controlled by `REVERSE_STEPS`.

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

## CIFAR-10 scheduler/prediction/loss-weight sweep

Run the 2x2x2 experiment matrix for:

- `SCHEDULER`: `linear`, `cosine`
- `PRED_TYPE`: `velocity`, `noise`
- `LOSS_WEIGHT_TYPE`: `constant`, `min_snr`

The automation script trains each split sequentially, generates 50,000 images
from the latest EMA epoch checkpoint, and computes FID against CIFAR-10 test
images. Training, image generation, and FID are run as separate subprocesses to
reduce split-to-split GPU memory carryover.

```bash
python scripts/run_cifar10_scheduler_pred_sweep.py
```

Use `--dry-run` to write the split configs and print the commands without
starting training:

```bash
python scripts/run_cifar10_scheduler_pred_sweep.py --dry-run
```

Example 2x2x2 CIFAR-10 run:

```bash
python ./scripts/run_cifar10_scheduler_pred_sweep.py \
  --base-config configs/cifar10_32x32_net01.yaml \
  --work-dir ./experiments/cifar10_scheduler_pred_losswt_sweep/01__net01_batch32 \
  --output-root ./training_outputs/cifar10_scheduler_pred_losswt_sweep/01__net01_batch32 \
  --train-batch-size 32 \
  --gen-batch-size 512
```

Results are written under the selected work directory, for example
`experiments/cifar10_scheduler_pred_losswt_sweep/01__net01_batch32/results/`.

### CIFAR-10 test results

The following results use unconditional training, batch size 32, 100 training
epochs, EMA epoch-100 checkpoints, 100 DDIM reverse steps, 50,000 generated
images, and 10,000 CIFAR-10 test images for the real FID reference. Lower FID
is better.

#### Network structures

| Field | net01 | net02 | net03 |
| --- | --- | --- | --- |
| Config | `configs/cifar10_32x32_net01.yaml` | `configs/cifar10_32x32_net02.yaml` | `configs/cifar10_32x32_net03.yaml` |
| Image | 32 |  |  |
| Channels | 3 |  |  |
| Block size | 1 |  |  |
| Res blocks | 2 |  |  |
| Norm groups | 32 |  |  |
| Base width | 64 |  |  |
| Multipliers | `[1, 2, 4]` |  |  |
| Attention | `[false, true, true]` |  |  |
| Mid attention | true |  |  |
| Heads | 1 |  |  |
| Embedding | positional |  |  |
| Dropout | 0.1 |  |  |
| Kernel | 3 |  |  |
| Cross attention | false |  |  |
| Classes | uncond |  |  |
| Skip | per_block |  |  |
| Trainable weights | 16M |  |  |

#### FID comparison

| Scheduler | Prediction | Loss weight | FID (net01) | FID (net02) | FID (net03) |
| --- | --- | --- | ---: | ---: | ---: |
| cosine | noise | min_snr | 16.507735 | TBD | TBD |
| cosine | noise | constant | 18.843318 | TBD | TBD |
| cosine | velocity | constant | 22.497827 | TBD | TBD |
| cosine | velocity | min_snr | 24.118977 | TBD | TBD |
| linear | noise | min_snr | 24.761688 | TBD | TBD |
| linear | noise | constant | 26.189926 | TBD | TBD |
| linear | velocity | constant | 34.320912 | TBD | TBD |
| linear | velocity | min_snr | 35.252986 | TBD | TBD |

Best `net01` split in this run: `cosine_noise_min_snr` with FID `16.507735`.

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

from __future__ import annotations

"""Plot SNR, clipped min-SNR, and loss weights for supported schedules."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fddim.diffusion_utils import DiffusionUtility


TIMESTEPS = 1000
MIN_SNR_GAMMA = 5.0


def _schedule_curves(scheduler: str) -> dict[str, np.ndarray]:
    diff_util = DiffusionUtility(
        scheduler=scheduler,
        timesteps=TIMESTEPS,
        reverse_steps=TIMESTEPS,
        pred_type="noise",
    )
    t = np.arange(1, TIMESTEPS + 1)
    alphas = diff_util.alphas[t]
    snr_t = alphas / (1.0 - alphas + 1e-8)
    min_snr = np.minimum(snr_t, MIN_SNR_GAMMA)
    noise_weights = np.divide(
        min_snr,
        snr_t,
        out=np.zeros_like(snr_t),
        where=snr_t != 0.0,
    )

    return {
        "t": t,
        "snr_t": snr_t,
        "min_snr": min_snr,
        "noise_weights": noise_weights,
        "velocity_weights": min_snr / (snr_t + 1.0),
    }


def _plot_scheduler_curves(scheduler: str, output_path: Path) -> None:
    curves = _schedule_curves(scheduler)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    fig.suptitle(f"{scheduler} schedule Min-SNR curves")

    axes[0].plot(curves["t"], curves["snr_t"], label="SNR(t)")
    axes[0].plot(curves["t"], curves["min_snr"], label=f"min(SNR(t), {MIN_SNR_GAMMA:g})")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("SNR")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].plot(curves["t"], curves["noise_weights"], label="noise: min_snr / SNR")
    axes[1].plot(
        curves["t"],
        curves["velocity_weights"],
        label="velocity: min_snr / (SNR + 1)",
    )
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("loss weight")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_scheduler_curves_csv(scheduler: str, output_path: Path) -> None:
    curves = _schedule_curves(scheduler)
    rows = np.column_stack(
        [
            curves["t"],
            curves["snr_t"],
            curves["min_snr"],
            curves["noise_weights"],
            curves["velocity_weights"],
        ]
    )
    np.savetxt(
        output_path,
        rows,
        delimiter=",",
        header="timestep,snr_t,min_snr,noise_loss_weight,velocity_loss_weight",
        comments="",
    )


def test_plot_snr_and_min_snr_loss_weight_curves() -> None:
    output_dir = Path(os.environ.get("FDDIM_TEST_PLOT_DIR", ROOT / "tests" / "generated"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for scheduler in ("linear", "cosine"):
        plot_path = output_dir / f"snr_min_snr_loss_weights_{scheduler}.png"
        csv_path = output_dir / f"snr_min_snr_loss_weights_{scheduler}.csv"

        _plot_scheduler_curves(scheduler, plot_path)
        _save_scheduler_curves_csv(scheduler, csv_path)

        assert plot_path.is_file()
        assert plot_path.stat().st_size > 0
        assert csv_path.is_file()
        assert csv_path.stat().st_size > 0

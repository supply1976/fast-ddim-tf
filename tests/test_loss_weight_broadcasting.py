from __future__ import annotations

"""Regression tests for timestep loss-weight broadcasting.

Keras image losses with ``reduction="none"`` return one scalar per pixel with
shape ``(batch, height, width)``. DiffusionModel then reduces that to
``(batch, 1, 1)`` before applying timestep weights. The weights must also be
``(batch, 1, 1)`` so each sample's loss is multiplied only by its own timestep
weight. A rank-4 weight tensor, ``(batch, 1, 1, 1)``, broadcasts against
``(batch, 1, 1)`` as ``(batch, batch, 1, 1)`` and cross-applies weights between
different samples.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fddim.diffusion_model import DiffusionModel


def _toy_pixel_loss() -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = 4
    height = 3
    width = 2
    channels = 3

    y_true = tf.reshape(
        tf.linspace(0.0, 1.0, batch_size * height * width * channels),
        [batch_size, height, width, channels],
    )
    y_pred = y_true * 0.25

    loss_fn = keras.losses.MeanSquaredError(reduction="none")
    pixel_loss = loss_fn(y_true, y_pred)
    sample_weights = tf.constant([1.0, 2.0, 4.0, 8.0], dtype=tf.float32)
    return pixel_loss, sample_weights


def _source_weighted_loss(pixel_loss: tf.Tensor, sample_weights: tf.Tensor) -> tf.Tensor:
    reduced_loss = tf.reduce_mean(pixel_loss, axis=[1, 2], keepdims=True)
    loss_weights = sample_weights[:, None, None]
    return tf.reduce_mean(reduced_loss * loss_weights)


def _intended_weighted_loss(pixel_loss: tf.Tensor, sample_weights: tf.Tensor) -> tf.Tensor:
    per_sample_loss = tf.reduce_mean(pixel_loss, axis=[1, 2])
    return tf.reduce_mean(per_sample_loss * sample_weights)


def test_loss_weight_shape_does_not_cross_broadcast_batch_samples() -> None:
    """Current weighting path should preserve one weighted scalar per sample."""
    pixel_loss, sample_weights = _toy_pixel_loss()

    reduced_loss = tf.reduce_mean(pixel_loss, axis=[1, 2], keepdims=True)
    source_weights = sample_weights[:, None, None]
    weighted_loss = reduced_loss * source_weights

    assert tuple(pixel_loss.shape) == (4, 3, 2)
    assert tuple(reduced_loss.shape) == (4, 1, 1)
    assert tuple(source_weights.shape) == (4, 1, 1)
    assert tuple(weighted_loss.shape) == (4, 1, 1)
    np.testing.assert_allclose(
        _source_weighted_loss(pixel_loss, sample_weights).numpy(),
        _intended_weighted_loss(pixel_loss, sample_weights).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_rank_four_loss_weights_would_cross_broadcast_batch_samples() -> None:
    """Document the old failure mode that mixed batch items during weighting."""
    pixel_loss, sample_weights = _toy_pixel_loss()

    reduced_loss = tf.reduce_mean(pixel_loss, axis=[1, 2], keepdims=True)
    legacy_weights = sample_weights[:, None, None, None]
    legacy_weighted_loss = reduced_loss * legacy_weights

    assert tuple(legacy_weighted_loss.shape) == (4, 4, 1, 1)
    assert not np.isclose(
        tf.reduce_mean(legacy_weighted_loss).numpy(),
        _intended_weighted_loss(pixel_loss, sample_weights).numpy(),
    )


def test_diffusion_model_constant_loss_weights_match_train_step_shape() -> None:
    """Constant weighting should return the shape expected by train_step."""
    model = SimpleNamespace(loss_weight_type="constant")
    timesteps = tf.constant([1, 7, 31, 100], dtype=tf.int32)

    weights = DiffusionModel._compute_loss_weights(model, timesteps)

    assert tuple(weights.shape) == (4, 1, 1)
    np.testing.assert_allclose(weights.numpy(), np.ones((4, 1, 1), dtype=np.float32))


def test_diffusion_model_min_snr_loss_weights_match_train_step_shape() -> None:
    """Min-SNR weighting should return per-sample weights without extra rank."""
    snr_values = np.array([100.0, 20.0, 5.0, 2.0, 0.5], dtype=np.float32)
    model = SimpleNamespace(
        loss_weight_type="min_snr",
        min_snr_gamma=5.0,
        snr_values=tf.constant(snr_values, dtype=tf.float32),
    )
    timesteps = tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)

    weights = DiffusionModel._compute_loss_weights(model, timesteps)

    expected = np.minimum(snr_values, 5.0) / (snr_values + 1e-8)
    assert tuple(weights.shape) == (5, 1, 1)
    np.testing.assert_allclose(weights.numpy(), expected[:, None, None], rtol=1e-6)

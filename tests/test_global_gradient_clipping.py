from __future__ import annotations

"""Regression tests for model-wide gradient clipping."""

import sys
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fddim.diffusion_model import DiffusionModel


class RecordingOptimizer:
    def __init__(self):
        self.grads_and_vars = None

    def apply_gradients(self, grads_and_vars):
        self.grads_and_vars = list(grads_and_vars)


def test_apply_gradients_clips_the_combined_global_norm():
    network = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(2, use_bias=True),
    ])
    model = SimpleNamespace(
        network=network,
        optimizer=RecordingOptimizer(),
        _update_ema_weights=lambda: None,
    )

    gradients = [
        tf.constant([[3.0, 0.0], [0.0, 0.0]]),
        tf.constant([4.0, 0.0]),
    ]

    DiffusionModel._apply_gradients(model, gradients)

    applied_grads = [gradient for gradient, _ in model.optimizer.grads_and_vars]
    tf.debugging.assert_near(tf.linalg.global_norm(applied_grads), 1.0)
    tf.debugging.assert_near(applied_grads[0][0, 0], 0.6)
    tf.debugging.assert_near(applied_grads[1][0], 0.8)


def test_apply_gradients_skips_disconnected_weights():
    network = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(2, use_bias=True),
    ])
    model = SimpleNamespace(
        network=network,
        optimizer=RecordingOptimizer(),
        _update_ema_weights=lambda: None,
    )

    DiffusionModel._apply_gradients(model, [tf.ones((2, 2)), None])

    assert len(model.optimizer.grads_and_vars) == 1
    assert model.optimizer.grads_and_vars[0][1] is model.network.trainable_weights[0]

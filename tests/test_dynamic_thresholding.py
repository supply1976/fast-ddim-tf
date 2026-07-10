from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fddim.diffusion_utils import DiffusionUtility


def test_dynamic_thresholding_scales_predicted_x0_per_sample() -> None:
    diff_util = DiffusionUtility(
        scheduler="linear",
        timesteps=10,
        reverse_steps=10,
        pred_type="image",
        clip_denoise=True,
        clip_denoise_mode="dynamic",
        dynamic_threshold_percentile=0.75,
    )
    x_t = tf.zeros((1, 2, 2, 1), dtype=tf.float32)
    t = tf.constant([1], dtype=tf.int32)
    y_pred = tf.constant([[[[0.0], [0.5]], [[2.0], [4.0]]]], dtype=tf.float32)

    pred_noise, pred_image = diff_util.get_pred_components(
        x_t, t, "image", y_pred, clip_denoise=True
    )

    expected_image = np.array([[[[0.0], [0.25]], [[1.0], [1.0]]]], dtype=np.float32)
    mu_t = tf.gather(diff_util.mu_coefs, t)[:, None, None, None]
    sigma_t = tf.gather(diff_util.sigma_coefs, t)[:, None, None, None]
    expected_noise = (x_t - mu_t * y_pred) / (sigma_t + 1.0e-8)

    np.testing.assert_allclose(pred_image.numpy(), expected_image, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(pred_noise.numpy(), expected_noise.numpy(), rtol=1.0e-6, atol=1.0e-6)
    assert np.max(np.abs(pred_image.numpy())) <= 1.0

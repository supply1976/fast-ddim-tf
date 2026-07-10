from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fddim.diffusion_utils import DiffusionUtility
from fddim.image_generator import ImageGenerator


class _StubImageGenerator(ImageGenerator):
    def __init__(self, diff_util: DiffusionUtility, pred_noise_s: tf.Tensor) -> None:
        super().__init__(
            diff_util,
            network=None,
            ema_network=None,
            timesteps=diff_util.timesteps,
        )
        self.pred_noise_s = pred_noise_s
        self.endpoint_evaluations = 0

    def _denoise_for_task(self, *args, **kwargs):
        self.endpoint_evaluations += 1
        x_t = args[0]
        return self.pred_noise_s, tf.zeros_like(x_t)


class _ZeroNoiseNetwork:
    def __call__(self, inputs, training=False):
        del training
        return tf.zeros_like(inputs[0])


def _make_diffusion_utility(
    *,
    threshold_mode: str = "fixed",
    ddim_eta: float = 0.0,
) -> DiffusionUtility:
    return DiffusionUtility(
        scheduler="cosine",
        timesteps=10,
        reverse_steps=5,
        pred_type="noise",
        ddim_eta=ddim_eta,
        clip_denoise_mode=threshold_mode,
        dynamic_threshold_percentile=0.75,
    )


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        (None, "ddim_1st"),
        ("ddim", "ddim_1st"),
        ("flow_euler", "ddim_1st"),
        ("ddim_1st", "ddim_1st"),
        ("flow_heun", "ddim_2nd"),
        ("ddim_2nd", "ddim_2nd"),
    ],
)
def test_sampler_aliases_have_consistent_canonical_names(alias, expected) -> None:
    generator = _StubImageGenerator(
        _make_diffusion_utility(), pred_noise_s=tf.constant(0.0)
    )

    assert generator._normalize_sampler(alias) == expected


def test_second_order_ddim_rejects_stochastic_eta() -> None:
    generator = _StubImageGenerator(
        _make_diffusion_utility(ddim_eta=0.5), pred_noise_s=tf.constant(0.0)
    )

    with pytest.raises(ValueError, match="ddim_2nd.*DDIM_ETA"):
        generator._normalize_sampler("ddim_2nd")


def test_first_order_ddim_matches_standard_ddim_update() -> None:
    diff_util = _make_diffusion_utility()
    x_t = tf.constant([[[[0.0], [0.5]], [[1.0], [2.0]]]], dtype=tf.float32)
    pred_noise_t = tf.ones_like(x_t) * 0.25
    pred_image_t = tf.clip_by_value(x_t, -1.0, 1.0)
    t = 8
    s = 6
    t_batch = tf.constant([t], dtype=tf.int32)
    s_batch = tf.constant([s], dtype=tf.int32)
    generator = _StubImageGenerator(diff_util, pred_noise_s=tf.zeros_like(x_t))

    actual = generator._sample_reverse_step(
        x_t,
        t,
        s,
        pred_noise_t,
        pred_image_t,
        sampler="ddim_1st",
        clip_denoise=True,
        labels=None,
    )
    expected = diff_util.p_sample_ddim(
        pred_image_t, pred_noise_t, t_batch, s_batch
    )

    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=1.0e-6, atol=1.0e-6
    )
    assert generator.endpoint_evaluations == 0


@pytest.mark.parametrize("threshold_mode", ["fixed", "dynamic"])
def test_second_order_ddim_thresholds_corrected_pred_image(
    threshold_mode: str,
) -> None:
    diff_util = _make_diffusion_utility(threshold_mode=threshold_mode)
    x_t = tf.constant([[[[0.0], [1.0]], [[2.0], [4.0]]]], dtype=tf.float32)
    pred_noise_t = tf.zeros_like(x_t)
    pred_noise_s = tf.ones_like(x_t) * 0.5
    pred_image_t = tf.zeros_like(x_t)
    t = 8
    s = 6
    t_batch = tf.constant([t], dtype=tf.int32)
    s_batch = tf.constant([s], dtype=tf.int32)
    generator = _StubImageGenerator(diff_util, pred_noise_s=pred_noise_s)

    actual = generator._sample_reverse_step(
        x_t,
        t,
        s,
        pred_noise_t,
        pred_image_t,
        sampler="ddim_2nd",
        clip_denoise=True,
        labels=None,
    )

    corrected_pred_noise = 0.5 * (pred_noise_t + pred_noise_s)
    mu_t = tf.gather(diff_util.mu_coefs, t_batch)[:, None, None, None]
    sigma_t = tf.gather(diff_util.sigma_coefs, t_batch)[:, None, None, None]
    corrected_pred_image = (
        x_t - sigma_t * corrected_pred_noise
    ) / (mu_t + 1.0e-8)
    corrected_pred_image = diff_util.apply_denoise_threshold(corrected_pred_image)
    expected = diff_util.p_sample_ddim(
        corrected_pred_image, corrected_pred_noise, t_batch, s_batch
    )

    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=1.0e-6, atol=1.0e-6
    )
    assert generator.endpoint_evaluations == 1


def test_second_order_ddim_without_threshold_matches_heun_flow_update() -> None:
    diff_util = _make_diffusion_utility()
    x_t = tf.constant([[[[0.0], [1.0]], [[2.0], [4.0]]]], dtype=tf.float32)
    pred_noise_t = tf.ones_like(x_t) * 0.25
    pred_noise_s = tf.ones_like(x_t) * 0.75
    t = 8
    s = 6
    t_batch = tf.constant([t], dtype=tf.int32)
    s_batch = tf.constant([s], dtype=tf.int32)
    mu_t = tf.gather(diff_util.mu_coefs, t_batch)[:, None, None, None]
    sigma_t = tf.gather(diff_util.sigma_coefs, t_batch)[:, None, None, None]
    pred_image_t = (x_t - sigma_t * pred_noise_t) / (mu_t + 1.0e-8)
    generator = _StubImageGenerator(diff_util, pred_noise_s=pred_noise_s)

    actual = generator._sample_reverse_step(
        x_t,
        t,
        s,
        pred_noise_t,
        pred_image_t,
        sampler="ddim_2nd",
        clip_denoise=False,
        labels=None,
    )
    expected = diff_util.p_sample_flow(
        x_t,
        pred_noise_t,
        t_batch,
        s_batch,
        pred_noise_s=pred_noise_s,
        order=2,
    )

    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=1.0e-6, atol=1.0e-6
    )


def test_second_order_ddim_uses_first_order_update_at_clean_endpoint() -> None:
    diff_util = _make_diffusion_utility()
    x_t = tf.ones((1, 2, 2, 1), dtype=tf.float32)
    pred_noise_t = tf.zeros_like(x_t)
    pred_image_t = tf.zeros_like(x_t)
    t = 2
    s = 0
    generator = _StubImageGenerator(diff_util, pred_noise_s=tf.ones_like(x_t))

    actual = generator._sample_reverse_step(
        x_t,
        t,
        s,
        pred_noise_t,
        pred_image_t,
        sampler="ddim_2nd",
        clip_denoise=True,
        labels=None,
    )
    expected = diff_util.p_sample_ddim(
        pred_image_t,
        pred_noise_t,
        tf.constant([t], dtype=tf.int32),
        tf.constant([s], dtype=tf.int32),
    )

    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=1.0e-6, atol=1.0e-6
    )
    assert generator.endpoint_evaluations == 0


@pytest.mark.parametrize("threshold_mode", ["fixed", "dynamic"])
def test_canvas_denoiser_thresholds_reconstructed_pred_image(
    threshold_mode: str,
) -> None:
    diff_util = _make_diffusion_utility(threshold_mode=threshold_mode)
    generator = ImageGenerator(
        diff_util,
        network=None,
        ema_network=_ZeroNoiseNetwork(),
        timesteps=diff_util.timesteps,
    )
    x_t = tf.constant([[[[0.0], [1.0]], [[2.0], [4.0]]]], dtype=tf.float32)
    t = tf.constant(5, dtype=tf.int32)

    pred_noise, pred_image = generator._denoise_step_patches_to_canvas(
        x_t,
        patch_size=2,
        stride=2,
        t=t,
        clip_denoise=True,
    )

    t_batch = tf.constant([5], dtype=tf.int32)
    mu_t = tf.gather(diff_util.mu_coefs, t_batch)[:, None, None, None]
    expected_image = diff_util.apply_denoise_threshold(x_t / (mu_t + 1.0e-8))
    np.testing.assert_allclose(pred_noise.numpy(), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        pred_image.numpy(), expected_image.numpy(), rtol=1.0e-6, atol=1.0e-6
    )

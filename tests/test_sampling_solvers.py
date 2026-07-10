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
    def __init__(self, image_shape=(2, 2, 1)):
        self.inputs = [
            tf.TensorSpec((None, *image_shape), tf.float32),
            tf.TensorSpec((None,), tf.int32),
        ]

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
        ("DPM-Solver++_2M", "dpmpp_2m"),
        ("dpmpp_2m", "dpmpp_2m"),
    ],
)
def test_sampler_aliases_have_consistent_canonical_names(alias, expected) -> None:
    generator = _StubImageGenerator(
        _make_diffusion_utility(), pred_noise_s=tf.constant(0.0)
    )

    assert generator._normalize_sampler(alias) == expected


@pytest.mark.parametrize("sampler", ["ddim_2nd", "dpmpp_2m"])
def test_second_order_samplers_reject_stochastic_eta(sampler: str) -> None:
    generator = _StubImageGenerator(
        _make_diffusion_utility(ddim_eta=0.5), pred_noise_s=tf.constant(0.0)
    )

    with pytest.raises(ValueError, match=f"{sampler}.*DDIM_ETA"):
        generator._normalize_sampler(sampler)


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


def test_dpmpp_2m_startup_matches_deterministic_ddim() -> None:
    diff_util = _make_diffusion_utility()
    generator = _StubImageGenerator(diff_util, pred_noise_s=tf.constant(0.0))
    state = generator._create_sampler_state("dpmpp_2m")
    x_t = tf.constant([[[[-1.0], [0.0]], [[1.0], [2.0]]]], dtype=tf.float32)
    pred_image_t = tf.constant(
        [[[[0.5], [-0.25]], [[0.75], [0.0]]]], dtype=tf.float32
    )
    t = 8
    s = 6
    mu_t = diff_util.mu_coefs[t]
    sigma_t = diff_util.sigma_coefs[t]
    pred_noise_t = (x_t - mu_t * pred_image_t) / sigma_t

    actual = generator._sample_reverse_step(
        x_t,
        t,
        s,
        pred_noise_t,
        pred_image_t,
        sampler="dpmpp_2m",
        clip_denoise=True,
        labels=None,
        sampler_state=state,
    )
    expected = diff_util.p_sample_ddim(
        pred_image_t,
        pred_noise_t,
        tf.constant([t], dtype=tf.int32),
        tf.constant([s], dtype=tf.int32),
    )

    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=1.0e-5, atol=1.0e-6
    )
    assert state.previous_timestep == t
    np.testing.assert_allclose(state.previous_pred_image.numpy(), pred_image_t.numpy())


def test_dpmpp_2m_matches_official_midpoint_multistep_equation() -> None:
    diff_util = _make_diffusion_utility()
    generator = _StubImageGenerator(diff_util, pred_noise_s=tf.constant(0.0))
    state = generator._create_sampler_state("dpmpp_2m")
    state.previous_timestep = 10
    state.previous_pred_image = tf.constant(
        [[[[0.0], [0.25]], [[0.5], [0.75]]]], dtype=tf.float32
    )
    x_t = tf.constant([[[[-1.0], [0.0]], [[1.0], [2.0]]]], dtype=tf.float32)
    pred_image_t = tf.constant(
        [[[[0.5], [0.0]], [[0.25], [1.0]]]], dtype=tf.float32
    )
    t = 8
    s = 5

    actual = generator._sample_reverse_step(
        x_t,
        t,
        s,
        pred_noise_t=tf.zeros_like(x_t),
        pred_image_t=pred_image_t,
        sampler="dpmpp_2m",
        clip_denoise=True,
        labels=None,
        sampler_state=state,
    )

    mu = diff_util.mu_coefs
    sigma = diff_util.sigma_coefs
    lambda_previous = tf.math.log(mu[10]) - tf.math.log(sigma[10])
    lambda_t = tf.math.log(mu[t]) - tf.math.log(sigma[t])
    lambda_s = tf.math.log(mu[s]) - tf.math.log(sigma[s])
    h_previous = lambda_t - lambda_previous
    h = lambda_s - lambda_t
    step_ratio = h_previous / h
    first_derivative = (
        pred_image_t
        - tf.constant([[[[0.0], [0.25]], [[0.5], [0.75]]]], dtype=tf.float32)
    ) / step_ratio
    phi_1 = tf.math.expm1(-h)
    expected = (
        (sigma[s] / sigma[t]) * x_t
        - mu[s] * phi_1 * pred_image_t
        - 0.5 * mu[s] * phi_1 * first_derivative
    )

    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=1.0e-6, atol=1.0e-6
    )


def test_dpmpp_2m_final_step_returns_current_data_prediction() -> None:
    diff_util = _make_diffusion_utility()
    generator = _StubImageGenerator(diff_util, pred_noise_s=tf.constant(0.0))
    state = generator._create_sampler_state("dpmpp_2m")
    state.previous_timestep = 4
    state.previous_pred_image = tf.zeros((1, 2, 2, 1), dtype=tf.float32)
    pred_image_t = tf.constant(
        [[[[0.5], [0.0]], [[-0.5], [1.0]]]], dtype=tf.float32
    )

    actual = generator._sample_reverse_step(
        tf.ones_like(pred_image_t),
        t=2,
        s=0,
        pred_noise_t=tf.zeros_like(pred_image_t),
        pred_image_t=pred_image_t,
        sampler="dpmpp_2m",
        clip_denoise=True,
        labels=None,
        sampler_state=state,
    )

    np.testing.assert_allclose(actual.numpy(), pred_image_t.numpy())


def test_dpmpp_2m_requires_persistent_state() -> None:
    diff_util = _make_diffusion_utility()
    generator = _StubImageGenerator(diff_util, pred_noise_s=tf.constant(0.0))
    x_t = tf.zeros((1, 2, 2, 1), dtype=tf.float32)

    with pytest.raises(ValueError, match="persistent sampler state"):
        generator._sample_reverse_step(
            x_t,
            t=8,
            s=6,
            pred_noise_t=tf.zeros_like(x_t),
            pred_image_t=tf.zeros_like(x_t),
            sampler="dpmpp_2m",
            clip_denoise=False,
            labels=None,
        )


@pytest.mark.parametrize("scheduler", ["linear", "cosine"])
def test_log_snr_timestep_spacing_is_strict_and_more_uniform(
    scheduler: str,
) -> None:
    diff_util = DiffusionUtility(
        scheduler=scheduler,
        timesteps=1000,
        reverse_steps=20,
        pred_type="velocity",
    )
    uniform_t, _ = diff_util.make_reverse_time_pairs(
        t_start=990, reverse_steps=20, timestep_spacing="uniform"
    )
    log_snr_t, log_snr_s = diff_util.make_reverse_time_pairs(
        t_start=990, reverse_steps=20, timestep_spacing="log_snr"
    )

    assert len(log_snr_t) == 20
    assert log_snr_t[0] == 990
    assert log_snr_s[-1] == 0
    assert np.all(log_snr_t > log_snr_s)
    assert np.all(np.diff(log_snr_t) < 0)

    mu = np.sqrt(diff_util.alphas)
    sigma = np.sqrt(1.0 - diff_util.alphas)
    half_log_snr = np.log(mu[1:]) - np.log(sigma[1:])
    uniform_intervals = np.diff(half_log_snr[uniform_t - 1])
    log_snr_intervals = np.diff(half_log_snr[log_snr_t - 1])
    assert np.std(log_snr_intervals) < np.std(uniform_intervals)


def test_reverse_timestep_spacing_rejects_unknown_value() -> None:
    diff_util = _make_diffusion_utility()

    with pytest.raises(ValueError, match="timestep_spacing"):
        diff_util.make_reverse_time_pairs(
            t_start=10,
            reverse_steps=5,
            timestep_spacing="karras",
        )


def test_sample_images_runs_dpmpp_2m_with_log_snr_spacing() -> None:
    diff_util = DiffusionUtility(
        scheduler="cosine",
        timesteps=10,
        reverse_steps=4,
        pred_type="velocity",
        ddim_eta=0.0,
    )
    network = _ZeroNoiseNetwork()
    generator = ImageGenerator(
        diff_util,
        network=network,
        ema_network=network,
        timesteps=diff_util.timesteps,
    )

    images = generator.sample_images(
        reverse_steps=4,
        num_images=2,
        sampler="dpmpp_2m",
        timestep_spacing="log_snr",
        clip_denoise=True,
    )

    assert images.shape == (2, 2, 2, 1)
    assert np.isfinite(images).all()
    assert np.all((0.0 <= images) & (images <= 1.0))

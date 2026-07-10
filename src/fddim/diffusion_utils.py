import numpy as np
import tensorflow as tf


class DiffusionUtility:
    """
    Utility class for diffusion model computations.
    
    Implements the mathematical operations for forward and reverse diffusion processes,
    supporting multiple noise schedules and prediction types.
    
    The forward process gradually adds noise: q(x_t | x_0) = N(√α_t x_0, (1-α_t)I)
    The reverse process removes noise: p(x_{t-n} | x_t) using learned predictions, n is the reverse stride.
    
    Always use time = 0.0 to 1.0, discretized into 'timesteps' intervals. so (timesteps + 1) points.
    
    Args:
        b0 (float): Starting beta value in linear schedule
        b1 (float): Ending beta value in linear schedule
        scheduler (str): Noise schedule type ('linear', 'cosine', 'my_cosine', 'my_cos6')
        timesteps (int): Number of diffusion timesteps
        pred_type (str): Type of model prediction ('noise', 'image', 'velocity')
        reverse_steps (int): Number of reverse steps in reverse process
        ddim_eta (float): DDIM determinism parameter (0=deterministic, 1=stochastic)
    """

    def __init__(
        self,
        b0=0.1,
        b1=20.0,
        scheduler='linear',
        timesteps=1000,
        pred_type='velocity',
        reverse_steps=1000,
        ddim_eta=0.0,
        clip_denoise=False,
        clip_denoise_mode='fixed',
        dynamic_threshold_percentile=0.995,
        dynamic_threshold_max=None,
    ):
        self._validate_params(timesteps, reverse_steps, pred_type)
        self._set_params(
            b0,
            b1,
            scheduler,
            timesteps,
            pred_type,
            reverse_steps,
            ddim_eta,
            clip_denoise_mode,
            dynamic_threshold_percentile,
            dynamic_threshold_max,
        )
        self.time_samples = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
        self.clip_denoise = clip_denoise
        
        # Compute diffusion coefficients
        self.alphas = self._compute_alphas()
        self._compute_forward_coefficients()
    
    def _validate_params(self, timesteps, reverse_steps, pred_type):
        """Validate initialization parameters."""
        if not isinstance(timesteps, int):
            raise TypeError("timesteps must be an integer")
        if not isinstance(reverse_steps, int):
            raise TypeError("reverse_steps must be an integer")
        if reverse_steps < 1:
            raise ValueError("reverse_steps must be >= 1")
        if reverse_steps > timesteps:
            raise ValueError("reverse_steps must be <= timesteps")
        if pred_type not in ['noise', 'image', 'velocity']:
            raise ValueError(f"pred_type must be one of ['noise', 'image', 'velocity'], got {pred_type}")
    
    def _set_params(
        self,
        b0,
        b1,
        scheduler,
        timesteps,
        pred_type,
        reverse_steps,
        ddim_eta,
        clip_denoise_mode,
        dynamic_threshold_percentile,
        dynamic_threshold_max,
    ):
        """Set instance parameters."""
        clip_denoise_mode = str(clip_denoise_mode).lower()
        if clip_denoise_mode not in ['fixed', 'dynamic']:
            raise ValueError("clip_denoise_mode must be one of ['fixed', 'dynamic']")
        if not 0.0 < float(dynamic_threshold_percentile) <= 1.0:
            raise ValueError("dynamic_threshold_percentile must be in (0, 1]")
        if dynamic_threshold_max is not None and float(dynamic_threshold_max) < 1.0:
            raise ValueError("dynamic_threshold_max must be None or >= 1.0")

        self.b0 = b0
        self.b1 = b1
        self.scheduler = scheduler
        self.timesteps = timesteps
        self.pred_type = pred_type
        self.reverse_steps = reverse_steps
        self.ddim_eta = ddim_eta
        self.CLIP_MIN = -1.0
        self.CLIP_MAX = 1.0
        self.clip_denoise_mode = clip_denoise_mode
        self.dynamic_threshold_percentile = float(dynamic_threshold_percentile)
        self.dynamic_threshold_max = (
            None if dynamic_threshold_max is None else float(dynamic_threshold_max)
        )
    
    def _compute_alphas(self):
        """Compute alpha values based on the chosen scheduler.
        alphas[0] = 1 (t=0)
        alphas[timesteps] --> 0 (t=1)
        returns: numpy array of shape (timesteps+1,)
        """
        if self.scheduler == 'linear':
            # beta(t) = b0 + t * (b1 - b0) for t in [0, 1]
            # Bt === integral of beta(t) dt from 0 to 1
            Bt = self.time_samples * self.b0 + 0.5 * self.time_samples**2 * (self.b1 - self.b0)
            if not np.all(Bt >= 0):
                raise ValueError("Beta values must be non-negative")
            return np.exp(-Bt)
        
        elif self.scheduler == 'my_cosine':
            end_angle = 89.99  # degrees
            angles = self.time_samples * end_angle * np.pi / 180
            return np.cos(angles) ** 2
        
        elif self.scheduler == 'cosine':
            # Original iDDPM cosine schedule (Nichol & Dhariwal, 2021).
            # Compute betas from alpha_bar, then convert to cumulative alphas.
            s = 0.008
            max_beta = 0.999

            def alpha_bar(t):
                return np.cos((t + s) / (1 + s) * np.pi / 2) ** 2

            betas = []
            for i in range(self.timesteps):
                t1 = i / self.timesteps
                t2 = (i + 1) / self.timesteps
                beta = 1 - alpha_bar(t2) / alpha_bar(t1)
                betas.append(min(beta, max_beta))

            betas = np.array(betas, dtype=np.float64)
            alphas = 1.0 - betas
            alpha_bars = np.cumprod(alphas, axis=0)
            return np.concatenate([[1.0], alpha_bars])
        
        elif self.scheduler == 'my_cos6':
            end_angle = 85.0  # degrees
            angles = self.time_samples * end_angle * np.pi / 180
            return np.cos(angles) ** 6
        
        else:
            raise ValueError(f"Unsupported scheduler '{self.scheduler}'. "
                           f"Supported: ['linear', 'cosine', 'my_cosine', 'my_cos6']")

    def _compute_forward_coefficients(self):
        """Compute coefficients for forward diffusion process."""
        mu_coefs = np.sqrt(self.alphas)
        var_coefs = 1.0 - self.alphas
        sigma_coefs = np.sqrt(var_coefs)
        # length = timesteps + 1
        # index 0 corresponds to t=0 (no noise), index timesteps corresponds to t=1 (full noise)
        self.mu_coefs = tf.constant(mu_coefs, tf.float32)
        self.var_coefs = tf.constant(var_coefs, tf.float32)
        self.sigma_coefs = tf.constant(sigma_coefs, tf.float32)
        self.alphas_tf = tf.constant(self.alphas, tf.float32)
    
    def _compute_reverse_coefficients(self, t, s):
        """Compute coefficients for reverse diffusion process.
        Args:
            t: Current timestep tensor or scalar
            s: Previous timestep tensor or scalar
            t > s
            Ex: t=10, s=9 for DDPM with 1 step reverse
                t=10, s=8 for DDPM with 2 step reverse
        """
        alpha_t = tf.gather(self.alphas_tf, t)
        alpha_s = tf.gather(self.alphas_tf, s)
        alpha_ts = alpha_t / (alpha_s + 1e-8)
        # Compute variance coefficients
        var_coefs_st = (1.0 - alpha_s) / (1.0 - alpha_t + 1e-8)
        reverse_var_coefs = var_coefs_st * (1.0 - alpha_ts)
        tf.debugging.assert_non_negative(reverse_var_coefs, message="Reverse variance coefficients must be non-negative")
        reverse_sigma_coefs = tf.sqrt(reverse_var_coefs)
        # Compute mean coefficients for DDPM and DDIM
        reverse_mu_ddpm_xt = var_coefs_st * tf.sqrt(alpha_ts)
        reverse_mu_ddpm_x0 = tf.sqrt(alpha_s) * (1.0 - alpha_ts) / (1.0 - alpha_t + 1e-8)
        reverse_mu_ddim_x0 = tf.sqrt(alpha_s)
        reverse_mu_ddim_noise = tf.sqrt(tf.maximum(1.0 - alpha_s - self.ddim_eta**2 * reverse_var_coefs, 0.0))
        self.reverse_sigma_coefs = reverse_sigma_coefs
        self.reverse_mu_ddpm_xt = reverse_mu_ddpm_xt
        self.reverse_mu_ddpm_x0 = reverse_mu_ddpm_x0
        self.reverse_mu_ddim_x0 = reverse_mu_ddim_x0
        self.reverse_mu_ddim_noise = reverse_mu_ddim_noise

    def q_sample(self, x_0, t, noise):
        sigma_t = tf.gather(self.sigma_coefs, t)[:, None, None, None]
        mu_t = tf.gather(self.mu_coefs, t)[:, None, None, None]
        x_t = mu_t * x_0 + sigma_t * noise
        v_t = mu_t * noise - sigma_t * x_0
        return (x_t, v_t)

    def make_reverse_time_pairs(
        self,
        t_start=None,
        reverse_steps=None,
        timestep_spacing='uniform',
    ):
        """Return strictly decreasing integer reverse timestep pairs.

        The returned arrays have length ``reverse_steps`` and represent
        transitions ``t_values[i] -> s_values[i]`` with ``t > s``. Uniform
        spacing is uniform in the model's discrete time labels. Log-SNR spacing
        chooses the positive model-evaluation timesteps approximately uniformly
        in ``lambda = log(mu) - log(sigma)`` and appends timestep zero only as
        the final update destination.
        """
        if t_start is None:
            t_start = self.timesteps
        if reverse_steps is None:
            reverse_steps = self.reverse_steps
        t_start = int(t_start)
        reverse_steps = int(reverse_steps)
        if t_start < 1 or t_start > self.timesteps:
            raise ValueError(f"t_start must be in range [1, {self.timesteps}]")
        if reverse_steps < 1:
            raise ValueError("reverse_steps must be >= 1")
        if reverse_steps > t_start:
            raise ValueError("reverse_steps must be <= t_start")

        spacing_aliases = {
            'uniform': 'uniform',
            'time_uniform': 'uniform',
            'log_snr': 'log_snr',
            'logsnr': 'log_snr',
        }
        spacing_key = str(timestep_spacing).strip().lower()
        if spacing_key not in spacing_aliases:
            raise ValueError(
                "timestep_spacing must be 'uniform' or 'log_snr'"
            )

        if spacing_aliases[spacing_key] == 'uniform':
            time_points = np.rint(
                np.linspace(t_start, 0, reverse_steps + 1, dtype=np.float64)
            ).astype(np.int32)
            time_points[0] = np.int32(t_start)
            time_points[-1] = np.int32(0)
        else:
            time_points = self._make_log_snr_time_points(t_start, reverse_steps)

        if np.any(np.diff(time_points) >= 0):
            raise ValueError("Reverse timestep grid must be strictly decreasing")
        return time_points[:-1], time_points[1:]

    def _make_log_snr_time_points(self, t_start, reverse_steps):
        """Map a uniform half-log-SNR grid onto discrete model timesteps."""
        if reverse_steps == 1:
            return np.array([t_start, 0], dtype=np.int32)

        mu = np.sqrt(self.alphas)
        sigma = np.sqrt(np.maximum(1.0 - self.alphas, 0.0))
        positive_timesteps = np.arange(t_start, 0, -1, dtype=np.int32)
        log_snr = (
            np.log(np.maximum(mu[positive_timesteps], 1.0e-30))
            - np.log(np.maximum(sigma[positive_timesteps], 1.0e-30))
        )
        target_log_snr = np.linspace(
            log_snr[0], log_snr[-1], reverse_steps, dtype=np.float64
        )
        interpolated_timesteps = np.interp(
            target_log_snr,
            log_snr,
            positive_timesteps.astype(np.float64),
        )
        model_timesteps = np.rint(interpolated_timesteps).astype(np.int32)
        model_timesteps[0] = np.int32(t_start)
        model_timesteps[-1] = np.int32(1)

        # Rounding can map adjacent log-SNR targets to the same label. Keep the
        # closest feasible label while reserving enough labels to reach 1.
        for index in range(1, reverse_steps - 1):
            largest = model_timesteps[index - 1] - 1
            smallest = reverse_steps - index
            model_timesteps[index] = np.clip(
                model_timesteps[index], smallest, largest
            )

        return np.concatenate(
            [model_timesteps, np.array([0], dtype=np.int32)]
        )

    def apply_denoise_threshold(self, pred_image):
        """Project predicted x0 using fixed clipping or dynamic thresholding."""
        if self.clip_denoise_mode == 'fixed':
            return tf.clip_by_value(pred_image, self.CLIP_MIN, self.CLIP_MAX)

        flat = tf.reshape(pred_image, (tf.shape(pred_image)[0], -1))
        abs_flat = tf.abs(flat)
        sorted_abs = tf.sort(abs_flat, axis=1)
        num_values = tf.shape(sorted_abs)[1]
        rank = tf.cast(
            tf.math.ceil(self.dynamic_threshold_percentile * tf.cast(num_values, tf.float32)),
            tf.int32,
        )
        rank = tf.clip_by_value(rank - 1, 0, num_values - 1)
        threshold = tf.gather(sorted_abs, rank, axis=1)
        threshold = tf.maximum(threshold, 1.0)
        if self.dynamic_threshold_max is not None:
            threshold = tf.minimum(threshold, self.dynamic_threshold_max)
        threshold = tf.reshape(threshold, (-1, 1, 1, 1))
        pred_image = tf.clip_by_value(pred_image, -threshold, threshold)
        return pred_image / threshold

    def get_pred_components(self, x_t, t, pred_type, y_pred, clip_denoise):
        """
        Convert model prediction to noise, image, and velocity components.
        
        Args:
            x_t: Noisy image at timestep t
            t: Timestep tensor
            pred_type: Type of prediction ('noise', 'image', 'velocity')
            y_pred: Model prediction
            clip_denoise: If True, project predicted clean image using
                ``clip_denoise_mode``.
                The model-implied noise is kept unchanged; recomputing it from
                clipped x0 can distort the deterministic DDIM/ODE direction.
        
        Returns:
            tuple: (pred_noise, pred_image)
        """
        # Get coefficients for timestep t
        var_t = tf.gather(self.var_coefs, t)[:, None, None, None]
        sigma_t = tf.gather(self.sigma_coefs, t)[:, None, None, None]
        mu_t = tf.gather(self.mu_coefs, t)[:, None, None, None]
        
        # Convert prediction based on type
        if pred_type == 'noise':
            pred_noise = y_pred
            pred_image = (x_t - sigma_t * pred_noise) / (mu_t + 1.0e-8)
            #pred_velocity = mu_t * pred_noise - sigma_t * pred_image
        elif pred_type == 'image':
            pred_image = y_pred
            pred_noise = (x_t - mu_t * pred_image) / (sigma_t + 1.0e-8)
            #pred_velocity = mu_t * pred_noise - sigma_t * pred_image
        elif pred_type == 'velocity':
            #pred_velocity = y_pred
            pred_image = mu_t * x_t - sigma_t * y_pred
            pred_noise = mu_t * y_pred + sigma_t * x_t
        else:
            raise ValueError(f"Invalid pred_type: {pred_type}")

        # Optional inference-time clipping: project only the denoised image.
        # Keep the model-implied noise as the reverse direction, especially for
        # deterministic DDIM and ODE solvers where eta=0.
        if clip_denoise:
            pred_image = self.apply_denoise_threshold(pred_image)

        return pred_noise, pred_image

    def p_sample_ddim(self, x_0, pred_noise, t, s):
        """Perform one step of DDIM reverse sampling.
        """
        self._compute_reverse_coefficients(t, s)
        rev_mu_ddim_x0 = self.reverse_mu_ddim_x0[:, None, None, None]
        rev_mu_ddim_noise = self.reverse_mu_ddim_noise[:, None, None, None]
        _mean = rev_mu_ddim_x0 * x_0 + rev_mu_ddim_noise * pred_noise
        _sigma = self.ddim_eta * self.reverse_sigma_coefs[:, None, None, None]
        if self.ddim_eta == 0.0:
            x_s = _mean
        else:
            noise = tf.random.normal(shape=_mean.shape, dtype=tf.float32)
            x_s = _mean + _sigma * noise
        return x_s

    def p_sample_flow(self, x_t, pred_noise_t, t, s, pred_noise_s=None, order=1):
        """Deterministic DDIM probability-flow ODE update.

        With ``order=1`` this is algebraically equivalent to deterministic
        DDIM (``eta=0``). With ``order=2`` it uses a trapezoidal/Heun correction
        in the ``x / sqrt(alpha)`` coordinate, evaluating the endpoint noise
        prediction separately and averaging the two noise fields.
        """
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if order == 2 and pred_noise_s is None:
            raise ValueError("pred_noise_s is required for order=2")

        mu_t = tf.gather(self.mu_coefs, t)[:, None, None, None]
        sigma_t = tf.gather(self.sigma_coefs, t)[:, None, None, None]
        mu_s = tf.gather(self.mu_coefs, s)[:, None, None, None]
        sigma_s = tf.gather(self.sigma_coefs, s)[:, None, None, None]

        eps = pred_noise_t
        if order == 2:
            eps = 0.5 * (pred_noise_t + pred_noise_s)

        gamma_t = sigma_t / (mu_t + 1.0e-8)
        gamma_s = sigma_s / (mu_s + 1.0e-8)
        y_t = x_t / (mu_t + 1.0e-8)
        y_s = y_t + (gamma_s - gamma_t) * eps
        return mu_s * y_s

    def get_timestep_info(self, t):
        """Get diffusion coefficients for a given timestep."""
        return {
            'mu': tf.gather(self.mu_coefs, t),
            'sigma': tf.gather(self.sigma_coefs, t),
            'var': tf.gather(self.var_coefs, t),
        }

    def validate_timestep(self, t):
        """Validate that timestep is within valid range."""
        if tf.reduce_any(t < 0) or tf.reduce_any(t > self.timesteps):
            raise ValueError(f"Timestep must be in range [0, {self.timesteps}]")

    @property
    def config(self):
        """Return configuration dictionary for serialization."""
        return {
            'b0': self.b0,
            'b1': self.b1,
            'scheduler': self.scheduler,
            'timesteps': self.timesteps,
            'pred_type': self.pred_type,
            'reverse_steps': self.reverse_steps,
            'ddim_eta': self.ddim_eta,
            'clip_denoise_mode': self.clip_denoise_mode,
            'dynamic_threshold_percentile': self.dynamic_threshold_percentile,
            'dynamic_threshold_max': self.dynamic_threshold_max,
        }


def unit_test_plot_alphas(output_path="alphas_schedules.png", timesteps=1000):
    """Plot alpha schedules for supported schedulers and save to PNG.

    Args:
        output_path (str): Destination PNG path.
        timesteps (int): Number of diffusion steps to visualize.
    """
    import matplotlib.pyplot as plt

    schedulers = ["linear", "cosine", "my_cosine", "my_cos6"]
    plt.figure(figsize=(8, 5))
    for scheduler in schedulers:
        util = DiffusionUtility(
            scheduler=scheduler,
            timesteps=timesteps,
            reverse_steps=timesteps,
            pred_type="velocity",
        )
        plt.plot(util.time_samples, util.alphas, label=scheduler)

    plt.title("Alpha Schedules")
    plt.xlabel("t")
    plt.ylabel("alpha_bar")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot diffusion alpha schedules")
    parser.add_argument("--output", default="alphas_schedules.png", help="Output PNG path")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion steps")
    args = parser.parse_args()

    unit_test_plot_alphas(output_path=args.output, timesteps=args.timesteps)

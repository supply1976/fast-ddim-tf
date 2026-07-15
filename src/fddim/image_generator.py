import gc
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image

from .patch_diffusion import append_coordinate_channels, coordinate_grid


@dataclass
class _DPMpp2MState:
    """History required by the second-order multistep data solver."""

    previous_pred_image: tf.Tensor | None = None
    previous_timestep: int | None = None


class ImageGenerator:
    """Run inference-only reverse diffusion for image generation tasks.

    The generator owns sampler selection, model evaluation, inference-time
    thresholding of predicted clean images, and task-specific conditioning. It
    supports first-order DDIM, a deterministic second-order DDIM/Heun
    corrector, and DPM-Solver++ 2M. All solvers work with random generation,
    patch-based canvas generation, image-to-image generation, and the inpainting
    variants handled by :meth:`generate_images_and_save`.

    Predicted noise is never modified by thresholding. When ``clip_denoise`` is
    enabled, only the predicted clean image (``pred_image`` or x0) is projected
    using the fixed or dynamic policy configured on ``diff_util``.

    Args:
        diff_util: Diffusion utility containing the schedule and DDIM update.
        network: Trained diffusion network.
        ema_network: Exponential-moving-average network used by default.
        timesteps: Number of forward diffusion timesteps.
        num_classes: Number of conditional classes, or ``None`` for an
            unconditional model.
    """

    _SAMPLER_ALIASES = {
        "ddim": "ddim_1st",
        "ddim_1st": "ddim_1st",
        "ddim_first": "ddim_1st",
        "ddim_euler": "ddim_1st",
        "euler": "ddim_1st",
        "flow_euler": "ddim_1st",
        "ddim_2nd": "ddim_2nd",
        "ddim_second": "ddim_2nd",
        "ddim_heun": "ddim_2nd",
        "heun": "ddim_2nd",
        "flow_heun": "ddim_2nd",
        "dpmpp_2m": "dpmpp_2m",
        "dpmpp2m": "dpmpp_2m",
        "dpm-solver++_2m": "dpmpp_2m",
        "dpm_solver++_2m": "dpmpp_2m",
        "dpmsolver++_2m": "dpmpp_2m",
        "dpm_solver_pp_2m": "dpmpp_2m",
    }
    
    def __init__(
        self,
        diff_util,
        network,
        ema_network,
        timesteps,
        num_classes=None,
        coordinate_conditioning=False,
    ):
        self.diff_util = diff_util
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.coordinate_conditioning = bool(coordinate_conditioning)

    def _normalize_sampler(self, sampler):
        """Return a canonical sampler name while accepting legacy aliases."""
        if sampler is None:
            sampler = "ddim_1st"
        if not isinstance(sampler, str):
            raise TypeError("sampler must be a string")

        sampler_key = sampler.strip().lower()
        if sampler_key not in self._SAMPLER_ALIASES:
            raise ValueError(
                "sampler must be 'ddim_1st', 'ddim_2nd', or 'dpmpp_2m' "
                "('ddim', 'flow_euler', and 'flow_heun' remain aliases)"
            )
        normalized_sampler = self._SAMPLER_ALIASES[sampler_key]
        if (
            normalized_sampler in {"ddim_2nd", "dpmpp_2m"}
            and self.diff_util.ddim_eta != 0.0
        ):
            raise ValueError(
                f"{normalized_sampler} is deterministic and requires "
                "DDIM_ETA: 0.0"
            )
        return normalized_sampler

    @staticmethod
    def _create_sampler_state(sampler):
        """Create per-batch history for samplers that require it."""
        if sampler == "dpmpp_2m":
            return _DPMpp2MState()
        return None

    def _threshold_pred_image(self, pred_image, enabled):
        """Apply the configured fixed or dynamic threshold to predicted x0."""
        if not enabled:
            return pred_image
        return self.diff_util.apply_denoise_threshold(pred_image)
    
    def _prepare_labels(self, num_images, labels=None):
        """Prepare class labels for conditional generation."""
        if labels is None and self.num_classes is not None:
            labels = tf.random.uniform((num_images,), minval=1, maxval=self.num_classes+1, dtype=tf.int32)
            #labels = tf.zeros((num_images,), dtype=tf.int32)
        return labels
    
    @tf.function
    def _denoise_step_patches_to_canvas(
        self,
        x_t,
        patch_size,
        stride,
        t,
        clip_denoise,
        labels=None,
    ):
        """Predict a canvas-wide noise field and x0 from overlapping patches.

        Patch noise predictions are averaged in overlap regions before the
        canvas-wide clean image is reconstructed. Thresholding is deliberately
        applied after reconstruction so dynamic thresholds are computed per
        complete output image rather than independently per patch.

        Args:
            x_t: Canvas tensor with shape ``(batch, height, width, channels)``.
            patch_size: Height and width of each square model input patch.
            stride: Spatial stride between overlapping patches.
            t: Scalar current timestep.
            clip_denoise: Whether to threshold the reconstructed x0.
            labels: Optional class labels for each canvas.

        Returns:
            A ``(pred_noise, pred_image)`` tuple matching the canvas shape.
        """
        # split x_t into patches
        patches = tf.image.extract_patches(
            images=x_t,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        ) # (B, n_h, n_w, patch_size*patch_size*C)
        batch_size = tf.shape(patches)[0]
        n_h = tf.shape(patches)[1]
        n_w = tf.shape(patches)[2]
        channels = tf.shape(x_t)[-1]
        patches = tf.reshape(
            patches, (-1, patch_size, patch_size, channels)
        )

        if self.coordinate_conditioning:
            canvas_coordinates = coordinate_grid(
                tf.shape(x_t)[1],
                tf.shape(x_t)[2],
                batch_size=batch_size,
                dtype=x_t.dtype,
            )
            coordinate_patches = tf.image.extract_patches(
                images=canvas_coordinates,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, stride, stride, 1],
                rates=[1, 1, 1, 1],
                padding='VALID',
            )
            coordinate_patches = tf.reshape(
                coordinate_patches, (-1, patch_size, patch_size, 2)
            )
            network_patches = tf.concat((patches, coordinate_patches), axis=-1)
        else:
            network_patches = patches
        
        # Prepare timesteps for all patches
        t_batch = tf.fill((tf.shape(patches)[0],), t)
        inputs = [network_patches, t_batch]
        
        # Fix: Properly replicate labels for all patches
        # Each batch element's label should be repeated n_h*n_w times
        if labels is not None:
            labels_expanded = tf.repeat(labels, n_h * n_w)
            inputs.append(labels_expanded)
            
        y_pred = self.ema_network(inputs, training=False)
        pred_noise, _ = self.diff_util.get_pred_components(
            patches,
            t_batch,
            self.diff_util.pred_type,
            y_pred,
            clip_denoise=False,
        )
        # calculate the scores
        sigma_t_batch = tf.gather(self.diff_util.sigma_coefs, t_batch)[
            :, None, None, None
        ]
        scores = -pred_noise / sigma_t_batch
        scores = tf.reshape(
            scores,
            (batch_size, n_h, n_w, patch_size, patch_size, channels),
        )
        # blend scores (average) to reconstruct full canvas score
        # Initialize canvas and count for averaging
        canvas_shape = (
            batch_size,
            (n_h - 1) * stride + patch_size,
            (n_w - 1) * stride + patch_size,
            channels,
        )
        score_canvas = tf.zeros(canvas_shape, dtype=scores.dtype)
        count = tf.zeros(canvas_shape, dtype=scores.dtype)
        
        # Accumulate patches using tf.while_loop (proper tf.function support)
        def accumulate_patches(i, score_canvas, count):
            def process_column(j, score_canvas, count):
                h_start = i * stride
                w_start = j * stride
                score_patch = scores[:, i, j, :, :, :] # (B, patch_size, patch_size, C)
                
                # Create padded version of patch to add to canvas
                pad_top = h_start
                pad_bottom = tf.shape(score_canvas)[1] - h_start - patch_size
                pad_left = w_start
                pad_right = tf.shape(score_canvas)[2] - w_start - patch_size
                # pad the patch to the canvas size
                patch_padded = tf.pad(score_patch, [
                    [0, 0],
                    [pad_top, pad_bottom],
                    [pad_left, pad_right],
                    [0, 0]
                ])
                score_canvas = score_canvas + patch_padded
                
                # Add 1 to count for this patch region
                ones = tf.ones_like(score_patch)
                count_padded = tf.pad(ones, [
                    [0, 0],
                    [pad_top, pad_bottom],
                    [pad_left, pad_right],
                    [0, 0]
                ])
                count = count + count_padded
                
                return j + 1, score_canvas, count
            
            j = tf.constant(0)
            _, score_canvas, count = tf.while_loop(
                lambda j, c, cnt: j < n_w,
                process_column,
                [j, score_canvas, count]
            )
            return i + 1, score_canvas, count
        
        i = tf.constant(0)
        _, score_canvas, count = tf.while_loop(
            lambda i, c, cnt: i < n_h,
            accumulate_patches,
            [i, score_canvas, count]
        )
        
        score_canvas = score_canvas / (count + 1e-8)  # Add epsilon to avoid division by zero, [B, H, W, C]
        
        # Use TensorFlow assertion instead of Python assert
        tf.debugging.assert_equal(
            tf.shape(score_canvas), 
            tf.shape(x_t),
            message="Reconstructed canvas shape mismatch"
        )
        sigma_t = tf.gather(self.diff_util.sigma_coefs, tf.fill((tf.shape(x_t)[0],), t))[:, None, None, None]
        var_t = sigma_t ** 2
        mu_t = tf.gather(self.diff_util.mu_coefs, tf.fill((tf.shape(x_t)[0],), t))[:, None, None, None]
        # reconstruct pred_noise and pred_image for full canvas
        pred_noise_canvas = - score_canvas * sigma_t
        pred_image_canvas = (x_t + var_t * score_canvas) / (mu_t + 1.0e-8)
        pred_image_canvas = self._threshold_pred_image(
            pred_image_canvas, enabled=clip_denoise
        )
        return pred_noise_canvas, pred_image_canvas
     
    @tf.function 
    def _denoise_step(self, x_t, t, clip_denoise, use_ema_model=True, labels=None):
        """Predict noise and x0 at ``t``, thresholding only the x0 estimate."""
        t_batch = tf.fill((tf.shape(x_t)[0],), t)
        network_images = (
            append_coordinate_channels(x_t)
            if self.coordinate_conditioning
            else x_t
        )
        inputs = [network_images, t_batch]
        if labels is not None:
            inputs.append(labels)
        y_pred = None
        if use_ema_model:
            y_pred = self.ema_network(inputs, training=False)
        else:
            y_pred = self.network(inputs, training=False)
        pred_noise, pred_image = self.diff_util.get_pred_components(
            x_t, t_batch, self.diff_util.pred_type, y_pred, clip_denoise=False
        )
        pred_image = self._threshold_pred_image(
            pred_image, enabled=clip_denoise
        )
        return pred_noise, pred_image

    def _denoise_for_task(
        self,
        x_t,
        t,
        clip_denoise,
        labels,
        gen_task="random",
        canvas_patch_size=None,
        canvas_stride=None,
        use_ema_model=True,
    ):
        """Evaluate the model through the denoiser required by ``gen_task``."""
        if gen_task == 'canvas_gen':
            return self._denoise_step_patches_to_canvas(
                x_t,
                canvas_patch_size,
                canvas_stride,
                tf.constant(t, dtype=tf.int32),
                clip_denoise,
                labels,
            )
        return self._denoise_step(
            x_t,
            tf.constant(t, dtype=tf.int32),
            clip_denoise,
            use_ema_model=use_ema_model,
            labels=labels,
        )

    def _sample_reverse_step(
        self,
        x_t,
        t,
        s,
        pred_noise_t,
        pred_image_t,
        sampler,
        clip_denoise,
        labels,
        gen_task="random",
        canvas_patch_size=None,
        canvas_stride=None,
        use_ema_model=True,
        sampler_state=None,
    ):
        """Advance one reverse-time step with the selected deterministic solver.

        ``ddim_1st`` performs the standard DDIM update from ``t`` to ``s``.
        ``ddim_2nd`` first predicts ``x_s`` with that update, evaluates the
        denoiser at the predicted endpoint, then applies a trapezoidal (Heun)
        correction by averaging the two predicted noise fields. The corrected
        clean-image estimate is reconstructed at ``t`` and passed through the
        same fixed/dynamic threshold policy as the first-order estimate.

        ``dpmpp_2m`` integrates the thresholded data prediction directly in
        half-log-SNR. It uses the current and previous ``pred_image`` values for
        a second-order multistep update, falling back to first order for startup
        and the final transition to timestep zero.

        The endpoint model evaluation is skipped for ``s == 0`` because the
        clean endpoint has no subsequent integration interval.
        """
        if sampler == "dpmpp_2m":
            if not isinstance(sampler_state, _DPMpp2MState):
                raise ValueError("dpmpp_2m requires persistent sampler state")
            return self._sample_dpmpp_2m_step(
                x_t,
                t,
                s,
                pred_image_t,
                sampler_state,
            )

        t_batch = tf.cast(tf.fill((tf.shape(x_t)[0],), t), tf.int32)
        s_batch = tf.cast(tf.fill((tf.shape(x_t)[0],), s), tf.int32)
        x_s_predictor = self.diff_util.p_sample_ddim(
            pred_image_t, pred_noise_t, t_batch, s_batch
        )
        if sampler == "ddim_1st" or int(s) == 0:
            return x_s_predictor

        pred_noise_s, _ = self._denoise_for_task(
            x_s_predictor,
            int(s),
            clip_denoise,
            labels,
            gen_task=gen_task,
            canvas_patch_size=canvas_patch_size,
            canvas_stride=canvas_stride,
            use_ema_model=use_ema_model,
        )
        corrected_pred_noise = 0.5 * (pred_noise_t + pred_noise_s)
        mu_t = tf.gather(self.diff_util.mu_coefs, t_batch)[:, None, None, None]
        sigma_t = tf.gather(self.diff_util.sigma_coefs, t_batch)[:, None, None, None]
        corrected_pred_image = (
            x_t - sigma_t * corrected_pred_noise
        ) / (mu_t + 1.0e-8)
        corrected_pred_image = self._threshold_pred_image(
            corrected_pred_image, enabled=clip_denoise
        )
        return self.diff_util.p_sample_ddim(
            corrected_pred_image,
            corrected_pred_noise,
            t_batch,
            s_batch,
        )

    def _sample_dpmpp_2m_step(
        self,
        x_t,
        t,
        s,
        pred_image_t,
        state,
    ):
        """Apply one midpoint-form DPM-Solver++ second-order multistep update.

        The implementation follows the official DPM-Solver++ 2M equation with
        ``lambda = log(mu) - log(sigma)``. The first step bootstraps with the
        first-order DPM-Solver++ update, which is deterministic DDIM. Timestep
        zero has infinite log-SNR, so the final update also uses first order and
        returns the current data prediction directly.
        """
        use_first_order = (
            state.previous_pred_image is None
            or state.previous_timestep is None
            or int(s) == 0
        )
        if use_first_order:
            x_s = self._sample_dpmpp_first_order(x_t, t, s, pred_image_t)
        else:
            if state.previous_timestep <= int(t):
                raise ValueError(
                    "dpmpp_2m history must contain a timestep greater than t"
                )

            lambda_previous = self._half_log_snr(state.previous_timestep)
            lambda_t = self._half_log_snr(t)
            lambda_s = self._half_log_snr(s)
            h_previous = lambda_t - lambda_previous
            h = lambda_s - lambda_t
            step_ratio = h_previous / h
            first_derivative = (
                pred_image_t - state.previous_pred_image
            ) / step_ratio

            sigma_t = tf.gather(self.diff_util.sigma_coefs, int(t))
            sigma_s = tf.gather(self.diff_util.sigma_coefs, int(s))
            mu_s = tf.gather(self.diff_util.mu_coefs, int(s))
            phi_1 = tf.math.expm1(-h)
            x_s = (
                (sigma_s / sigma_t) * x_t
                - mu_s * phi_1 * pred_image_t
                - 0.5 * mu_s * phi_1 * first_derivative
            )

        state.previous_pred_image = pred_image_t
        state.previous_timestep = int(t)
        return x_s

    def _sample_dpmpp_first_order(self, x_t, t, s, pred_image_t):
        """Apply the first-order DPM-Solver++ update from ``t`` to ``s``."""
        if int(s) == 0:
            return pred_image_t

        lambda_t = self._half_log_snr(t)
        lambda_s = self._half_log_snr(s)
        h = lambda_s - lambda_t
        sigma_t = tf.gather(self.diff_util.sigma_coefs, int(t))
        sigma_s = tf.gather(self.diff_util.sigma_coefs, int(s))
        mu_s = tf.gather(self.diff_util.mu_coefs, int(s))
        return (
            (sigma_s / sigma_t) * x_t
            - mu_s * tf.math.expm1(-h) * pred_image_t
        )

    def _half_log_snr(self, timestep):
        """Return ``log(mu_t) - log(sigma_t)`` for a positive timestep."""
        timestep = int(timestep)
        if timestep <= 0:
            raise ValueError("half-log-SNR is finite only for timesteps > 0")
        mu_t = tf.gather(self.diff_util.mu_coefs, timestep)
        sigma_t = tf.gather(self.diff_util.sigma_coefs, timestep)
        return tf.math.log(mu_t) - tf.math.log(sigma_t)
    
    def _prepare_overlap_inpaint(self, base_images, overlap_dir, overlap_size):
        """
        Prepare base anchor and inpaint mask for overlap_inpaint task under adjacency semantics.

        Semantics:
          We generate a NEW tile of the same shape as base_images. This new tile is intended
          to be placed adjacent to the base along overlap_dir. The two tiles share an overlap
          strip of width/height = overlap_size which must remain identical (copied from base).

          For east: generated tile sits to the RIGHT of base; its LEFT overlap_size columns must
          equal base RIGHT overlap_size columns and are frozen (mask=0). Remaining columns are
          inpaint region (mask=1).
          For west: generated tile sits to the LEFT of base; its RIGHT overlap_size columns link
          to base LEFT overlap_size columns.
          For north: generated tile sits ABOVE base; its BOTTOM overlap_size rows match base TOP.
          For south: generated tile sits BELOW base; its TOP overlap_size rows match base BOTTOM.

        Returns:
          anchor_images: array with overlapped strip copied from base, elsewhere zeros (in [-1,1])
          inpaint_mask: 1 where we will generate, 0 where we keep anchor (overlap strip)
        """
        if base_images is None:
            raise ValueError("base_images must be provided for overlap_inpaint")
        if overlap_dir not in ['north','south','east','west']:
            raise ValueError(f"Invalid overlap_dir {overlap_dir}")
        if overlap_size is None or overlap_size <= 0:
            raise ValueError("overlap_size must be positive")

        anchor = np.zeros_like(base_images)
        mask = np.ones_like(base_images)
        H = base_images.shape[1]
        W = base_images.shape[2]
        if overlap_dir in ['north','south'] and overlap_size >= H:
            raise ValueError("overlap_size must be < image height for north/south overlap")
        if overlap_dir in ['east','west'] and overlap_size >= W:
            raise ValueError("overlap_size must be < image width for east/west overlap")

        if overlap_dir == 'east':
            # copy right strip of base into left strip of generated tile
            anchor[:, :, 0:overlap_size, :] = base_images[:, :, -overlap_size:, :]
            mask[:, :, 0:overlap_size, :] = 0
        elif overlap_dir == 'west':
            anchor[:, :, -overlap_size:, :] = base_images[:, :, 0:overlap_size, :]
            mask[:, :, -overlap_size:, :] = 0
        elif overlap_dir == 'north':
            anchor[:, -overlap_size:, :, :] = base_images[:, 0:overlap_size, :, :]
            mask[:, -overlap_size:, :, :] = 0
        elif overlap_dir == 'south':
            anchor[:, 0:overlap_size, :, :] = base_images[:, -overlap_size:, :, :]
            mask[:, 0:overlap_size, :, :] = 0
        return anchor, mask
    
    def sample_images(
        self,
        reverse_steps=100,
        num_images=20,
        clip_denoise=True,
        use_ema_model=True,
        labels=None,
        sampler="ddim_1st",
        t_start=None,
        timestep_spacing="uniform",
    ):
        """Generate an in-memory batch of random images.
        
        This is a lightweight variant used for inline evaluation where we only
        need the final samples rather than saving them to disk.
        
        Args:
            reverse_steps: Number of reverse-time transitions.
            num_images: Number of images to generate.
            clip_denoise: Whether to threshold each predicted clean image using
                the fixed or dynamic policy configured on ``diff_util``.
            use_ema_model: Whether to evaluate the EMA network.
            labels: Optional class labels for conditional generation.
            sampler: ``"ddim_1st"``, ``"ddim_2nd"``, or ``"dpmpp_2m"``.
                Legacy aliases are accepted.
            t_start: Starting timestep, or ``None`` for ``self.timesteps``.
            timestep_spacing: ``"uniform"`` for uniform discrete timesteps or
                ``"log_snr"`` for approximately uniform half-log-SNR steps.
            
        Returns:
            A float32 NumPy array in ``[0, 1]``.
        """
        sampler = self._normalize_sampler(sampler)
        # prepare initial samples and labels
        img_h, img_w = self.network.inputs[0].shape[1:3]
        img_c = (
            self.network.output_shape[-1]
            if hasattr(self.network, "output_shape")
            else self.network.inputs[0].shape[-1]
            - (2 if self.coordinate_conditioning else 0)
        )
        shape = (num_images, img_h, img_w, img_c)
        samples = tf.random.normal(shape=shape, dtype=tf.float32) 
        labels = self._prepare_labels(num_images, labels)
        if t_start is None:
            t_start = self.timesteps
        reverse_timeindex, reverse_nextindex = self.diff_util.make_reverse_time_pairs(
            t_start, reverse_steps, timestep_spacing
        )
        sampler_state = self._create_sampler_state(sampler)
         
        for t, s in tqdm.tqdm(zip(reverse_timeindex, reverse_nextindex), total=len(reverse_timeindex)):
            pred_noise_t, pred_image_t = self._denoise_step(
                samples, tf.constant(t, dtype=tf.int32), clip_denoise, use_ema_model, labels
            )
            samples = self._sample_reverse_step(
                samples,
                int(t),
                int(s),
                pred_noise_t,
                pred_image_t,
                sampler,
                clip_denoise,
                labels,
                use_ema_model=use_ema_model,
                sampler_state=sampler_state,
            )
        # final postprocessing
        samples = samples.numpy()
        samples = np.clip(samples, -1, 1)
        samples = 0.5 * (samples + 1) # to [0,1], float32
        return samples
    
    def generate_images_and_save(self,
                                 logs=None,
                                 gen_task="random",
                                 num_images=20,
                                 canvas_shape=None,
                                 canvas_patch_size=None,
                                 canvas_stride=None,
                                 reverse_steps=100,
                                 sampler="ddim_1st",
                                 t_start=None,
                                 timestep_spacing="uniform",
                                 savedir='./',
                                 save_intermediate=False,
                                 save_format='png',
                                 clip_denoise=True, 
                                 base_images=None,
                                 labels=None,
                                 inpaint_mask=None,
                                 freeze_channel=None,
                                 space_inpaint_bbox=None,
                                 bbox_to_inpaint=True,
                                 self_guide_scale=0.0,
                                 sdedit_strength=None,
                                 _renoise_base_images=True,
                                 overlap_dir=None,
                                 overlap_size=None,
                                 batch_size=None,
                                 images_per_subfolder=10000,
                                 ):
        """
        Generate images with optional intermediate saving and progress tracking.
        Supports large-scale generation (e.g., 50k images) with batching and subfolder organization.
        
        Args:
            gen_task: Task type for generation (e.g., 'random', 'channel_inpaint', 'space_inpaint', 'overlap_inpaint')
            num_images: Total number of images to generate
            canvas_shape: For large canvas generation, tuple of (height, width)
            canvas_patch_size: For large canvas generation, size of each patch
            canvas_stride: For large canvas generation, stride between patches
            reverse_steps: Number of reverse steps for diffusion
            sampler: ``"ddim_1st"``, ``"ddim_2nd"``, or ``"dpmpp_2m"``.
                Legacy aliases are accepted.
            t_start: Optional starting timestep for random generation. If None,
                starts from the scheduler's final timestep.
            timestep_spacing: ``"uniform"`` or ``"log_snr"``.
            savedir: Directory to save generated images
            save_intermediate: Whether to save denoised images at intermediate timesteps
            clip_denoise: Whether to apply the configured fixed or dynamic
                threshold to predicted clean images during reverse sampling.
            base_images: Optional base images for inpainting
            labels: Optional class labels
            inpaint_mask: Optional mask for inpainting
            freeze_channel: Optional channel to freeze for inpainting
            space_inpaint_bbox: Optional bounding box for space inpainting
            self_guide_scale: Scale for self-guidance during generation
            save_format: Format to save images (png or npz)
            overlap_dir: Direction for overlap adjacency ('north','south','east','west'). Generated tile is intended to sit in this direction relative to base; the overlapping strip is copied from base and frozen.
            overlap_size: Size (pixels) of the shared overlap strip. For east/west this is column width; for north/south this is row height.
            batch_size: Number of images to generate per batch (None for auto-selection based on num_images)
            images_per_subfolder: Number of images to save per subfolder (for large-scale generation)

        Returns:
        """
        sampler = self._normalize_sampler(sampler)
        if sampler != "ddim_1st" and self_guide_scale > 0.0:
            raise ValueError(
                "SELF_GUIDE_SCALE is only supported with sampler='ddim_1st'"
            )
        if batch_size is None:
            batch_size = min(100, num_images)
        logging.info(f"[IMGEN] ===== Image Generation =====")
        logging.info(f"[IMGEN] Total images to generate: {num_images}")
        logging.info(f"[IMGEN] Batch size: {batch_size}")
        logging.info(f"[IMGEN] Number of batches: {(num_images + batch_size - 1) // batch_size}")
        logging.info(f"[IMGEN] Images per subfolder: {images_per_subfolder}")
        logging.info(f"[IMGEN] Reverse steps: {reverse_steps}")
        logging.info(f"[IMGEN] Sampler: {sampler}")
        logging.info(f"[IMGEN] Timestep spacing: {timestep_spacing}")
        logging.info(f"[IMGEN] T start: {t_start if t_start is not None else self.timesteps}")
        
        # Create main save directory
        Path(savedir).mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        total_generated = 0
        batch_count = 0
        
        # Generate images in batches
        for batch_start in range(0, num_images, batch_size):
            batch_end = min(batch_start + batch_size, num_images)
            current_batch_size = batch_end - batch_start
            batch_count += 1
            if labels is not None:
                batch_labels = labels[batch_start:batch_end]
            if base_images is not None:
                batch_base_images = base_images[batch_start:batch_end]
            logging.info(f"[IMGEN] ----- Batch {batch_count}: Generating images {batch_start} to {batch_end-1} -----")
            # Generate batch
            try:
                batch_output = self._generate_single_batch(
                    batch_size=current_batch_size,
                    batch_start_idx=batch_start,
                    gen_task=gen_task,
                    canvas_shape=canvas_shape,
                    canvas_patch_size=canvas_patch_size,
                    canvas_stride=canvas_stride,
                    reverse_steps=reverse_steps,
                    sampler=sampler,
                    t_start=t_start,
                    timestep_spacing=timestep_spacing,
                    savedir=savedir,
                    save_intermediate=save_intermediate,
                    save_format=save_format,
                    clip_denoise=clip_denoise,
                    base_images=batch_base_images if base_images is not None else None,
                    labels=batch_labels if labels is not None else None,
                    inpaint_mask=inpaint_mask,
                    freeze_channel=freeze_channel,
                    space_inpaint_bbox=space_inpaint_bbox,
                    bbox_to_inpaint=bbox_to_inpaint,
                    self_guide_scale=self_guide_scale,
                    sdedit_strength=sdedit_strength,
                    _renoise_base_images=_renoise_base_images,
                    overlap_dir=overlap_dir,
                    overlap_size=overlap_size,
                    images_per_subfolder=images_per_subfolder,
                )
                
                total_generated += current_batch_size
                logging.info(f"[IMGEN] Batch {batch_count} completed. Progress: {total_generated}/{num_images}")
                sys.stdout.flush()
                
            except Exception as e:
                logging.error(f"[IMGEN] Error in batch {batch_count}: {str(e)}")
                logging.exception(e)
                raise
            
            finally:
                # Clean up memory after each batch
                self._cleanup_memory()
            
            self._save_batch_images(
                batch_output['final'],
                batch_start,
                savedir,
                save_format,
                images_per_subfolder,
            )
            
            # Save intermediate denoised images if enabled
            if save_intermediate and 'intermediate' in batch_output:
                intermediate_dir = Path(savedir) / 'intermediate'
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                for step_key, step_images in batch_output['intermediate'].items():
                    step_dir = intermediate_dir / step_key
                    step_dir.mkdir(parents=True, exist_ok=True)
                    self._save_batch_images(
                        step_images,
                        batch_start,
                        str(step_dir),
                        save_format,
                        images_per_subfolder,
                    )
        
        logging.info(f"[IMGEN] ===== Generation Complete =====")
        logging.info(f"[IMGEN] Total images generated: {total_generated}")
        logging.info(f"[IMGEN] All images saved to: {savedir}")
        
        return {
            'total_generated': total_generated,
            'num_batches': batch_count,
            'savedir': savedir,
        }
    
    def _cleanup_memory(self):
        """Clean up memory after batch generation."""
        # Avoid clearing the global TensorFlow session between batches; this can break model state
        gc.collect()
        logging.debug("[IMGEN] Memory cleanup completed")
    
    def _generate_single_batch(self,
                               batch_size,
                               batch_start_idx,
                               gen_task,
                               canvas_shape,
                               canvas_patch_size,
                               canvas_stride,
                               reverse_steps,
                               sampler,
                               t_start,
                               timestep_spacing,
                               savedir,
                               save_intermediate,
                               save_format,
                               clip_denoise,
                               base_images,
                               labels,
                               inpaint_mask,
                               freeze_channel,
                               space_inpaint_bbox,
                               bbox_to_inpaint,
                               self_guide_scale,
                               sdedit_strength,
                               _renoise_base_images,
                               overlap_dir,
                               overlap_size,
                               images_per_subfolder,
                               ):
        """
        Generate a single batch of images.
        
        This is the core generation logic extracted from the original generate_images_and_save
        to enable batch processing for large-scale generation.
        """
        img_h, img_w = self.network.inputs[0].shape[1:3]
        img_c = (
            self.network.output_shape[-1]
            if hasattr(self.network, "output_shape")
            else self.network.inputs[0].shape[-1]
            - (2 if self.coordinate_conditioning else 0)
        )
        
        # Handle canvas generation
        if gen_task == 'canvas_gen':
            assert canvas_shape is not None
            assert canvas_patch_size is not None
            assert canvas_stride is not None
            assert canvas_shape[0] >= img_h and canvas_shape[1] >= img_w
            img_h, img_w = canvas_shape
            if canvas_patch_size > img_h or canvas_patch_size > img_w:
                raise ValueError("canvas_patch_size must be <= canvas height/width")
            if (img_h - canvas_patch_size) % canvas_stride != 0 or (img_w - canvas_patch_size) % canvas_stride != 0:
                raise ValueError(
                    "canvas_shape must satisfy (H - canvas_patch_size) % canvas_stride == 0 "
                    "and (W - canvas_patch_size) % canvas_stride == 0 for full coverage"
                )

        # Initialize samples and labels for this batch
        samples = tf.random.normal(shape=(batch_size, img_h, img_w, img_c), dtype=tf.float32)
        batch_labels = self._prepare_labels(batch_size, labels)
        
        # Handle different generation tasks
        if t_start is None:
            t_start = self.timesteps
        t_start = int(t_start)
        if gen_task == 'img2img':
            assert base_images is not None
            if t_start == self.timesteps:
                t_start = int(self.timesteps * sdedit_strength) if sdedit_strength is not None else int(self.timesteps * 0.5)
            samples, _ = self.diff_util.q_sample(base_images, tf.fill((batch_size,), t_start), samples)
        
        if gen_task == 'channel_inpaint':
            assert base_images is not None
            assert freeze_channel is not None
            # freeze_channel can be int or list of ints for multiple channels
            if isinstance(freeze_channel, int):
                freeze_channel = [freeze_channel]
            for ch in freeze_channel:
                if ch < 0 or ch >= img_c:
                    raise ValueError(f"freeze_channel must be in range [0, {img_c - 1}]")
            # Create a mask that zeros out the specified channels in the generated samples, effectively freezing them to the base image values
            # mask=1 for inpainting (re-generate), mask=0 to keep the observed (base_images)
            inpaint_mask = np.ones_like(base_images)
            inpaint_mask[..., freeze_channel] = 0

        if gen_task == 'space_inpaint':
            assert base_images is not None
            assert space_inpaint_bbox is not None
            x0,y0,x1,y1 = space_inpaint_bbox
            # create a binary mask where the region to inpaint is 1 and the rest is 0 (or vice versa based on bbox_to_inpaint)
            # mask=1 for inpainting (re-generate), mask=0 to keep the observed (base_images)
            inpaint_mask = np.zeros_like(base_images)
            inpaint_mask[:, x0:x1, y0:y1, :] = 1
            if not bbox_to_inpaint:
                inpaint_mask = 1 - inpaint_mask
                
        if gen_task == 'overlap_inpaint':
            # adjacency semantics: copy overlap strip from base into generated tile anchor
            anchor_images, inpaint_mask = self._prepare_overlap_inpaint(base_images, overlap_dir, overlap_size)
            # Replace base_images with anchor_images for consistency in later masking logic
            base_images = anchor_images
            # (Optional) quick assertion: ensure overlap region identical
            # east example: left strip of anchor equals right strip of original base
            # We skip heavy checks for performance; users can enable in debug mode.

        if inpaint_mask is not None:
            inpaint_mask = tf.convert_to_tensor(inpaint_mask)
            base_images = tf.convert_to_tensor(base_images)
            # create a fixed noise0 for inpainting
            noise0 = tf.random.normal(shape=(batch_size, img_h, img_w, img_c), dtype=tf.float32)
            
        # Prepare output dictionary
        output_dict = {}
        if save_intermediate:
            output_dict['intermediate'] = {}
            logging.info("[IMGEN] Intermediate saving enabled - will store denoised images at each step")
        
        # Generation loop
        reverse_timeindex, reverse_nextindex = self.diff_util.make_reverse_time_pairs(
            t_start, reverse_steps, timestep_spacing
        )
        previous_pred_noise = None
        sampler_state = self._create_sampler_state(sampler)
        
        logging.debug(f"[IMGEN] Starting reverse diffusion with {len(reverse_timeindex)} steps")
        
        for t, s in list(zip(reverse_timeindex, reverse_nextindex)):
            # main loop for reverse diffusion
            # ex: 
            # t = [1000, 990, 980, ..., 10] for stride=10, steps=100
            # s = [990, 980, ..., 0]
            t_batch = tf.cast(
                tf.fill((tf.shape(samples)[0],), t), tf.int32
            )
            
            if inpaint_mask is not None:
                """
                # _renoise_base_image should be a hidden config for debugging purpose only, 
                # as it adds extra compute cost but can improve generation quality for inpainting tasks by 
                # maintaining data consistency at each step. 
                # We can consider exposing it as a user config later if needed.
                # re-noise the base_image at time t to make data consistency for inpainting
                # This is a hack to address the issue that for inpainting tasks, 
                # the base image is only added at the beginning (t_start), but as we go through reverse diffusion, 
                # the samples may deviate significantly from the base image distribution, causing poor generation quality. 
                # By re-noising the base image at each step, 
                # we ensure that the inpainted regions are always conditioned on a noisy version of 
                # the base image that matches the current timestep, which can improve stability and quality of the generated images.
                # reference paper: https://arxiv.org/abs/2201.09865 "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"
                """
                if _renoise_base_images:
                    base_images_t, _ = self.diff_util.q_sample(
                        base_images, t_batch, noise0
                    )
                    samples = samples * inpaint_mask + base_images_t * (1 - inpaint_mask)
                else:
                    samples = samples * inpaint_mask + base_images * (1 - inpaint_mask)
            
            pred_noise_t, pred_image_t = self._denoise_for_task(
                samples,
                int(t),
                clip_denoise,
                batch_labels,
                gen_task=gen_task,
                canvas_patch_size=canvas_patch_size,
                canvas_stride=canvas_stride,
                use_ema_model=True,
            )
            
            if self_guide_scale > 0.0:
                guided_pred_noise = (
                    pred_noise_t
                    + self_guide_scale * (previous_pred_noise - pred_noise_t)
                    if previous_pred_noise is not None
                    else pred_noise_t
                )
                s_batch = tf.cast(
                    tf.fill((tf.shape(samples)[0],), s), tf.int32
                )
                samples = self.diff_util.p_sample_ddim(
                    pred_image_t, guided_pred_noise, t_batch, s_batch
                )
                previous_pred_noise = pred_noise_t
            else:
                samples = self._sample_reverse_step(
                    samples,
                    int(t),
                    int(s),
                    pred_noise_t,
                    pred_image_t,
                    sampler,
                    clip_denoise,
                    batch_labels,
                    gen_task=gen_task,
                    canvas_patch_size=canvas_patch_size,
                    canvas_stride=canvas_stride,
                    use_ema_model=True,
                    sampler_state=sampler_state,
                )

            if s==0 and inpaint_mask is not None:
                # Final step correction to ensure exact data consistency at the end of generation for inpainting tasks.
                samples = samples * inpaint_mask + base_images * (1 - inpaint_mask)
            
            # save intermediate results (optional - typically not used for large-scale generation)
            if save_intermediate:
                intermediate_images = samples.numpy()
                intermediate_images = np.clip(intermediate_images, -1, 1)
                intermediate_images = 0.5 * (intermediate_images + 1)
                output_dict['intermediate'][f'step_{t:04d}'] = intermediate_images
        
        # Final postprocessing
        samples = samples.numpy()
        samples = np.clip(samples, -1, 1)
        samples = 0.5 * (samples + 1)
        output_dict['final'] = samples
        return output_dict
    
    def _save_batch_images(self, images, start_idx, savedir, save_format, images_per_subfolder):
        """
        Save a batch of images with proper subfolder organization.
        
        Args:
            images: Numpy array of images to save
            start_idx: Starting index for image numbering
            savedir: Root directory to save images
            save_format: Format to save images ('png', 'npz', etc.)
            images_per_subfolder: Number of images to save per subfolder
        """
        if save_format == 'npz':
            # For npz, save entire batch in a single file within a subfolder
            subfolder_idx = (start_idx // images_per_subfolder)*images_per_subfolder
            subfolder = Path(savedir) / f"sub_{subfolder_idx:06d}"
            subfolder.mkdir(parents=True, exist_ok=True)
            
            shape_str = "x".join(map(str, images.shape))
            eta = str(self.diff_util.ddim_eta)
            revs = str(self.diff_util.reverse_steps)
            filename = f"images_{start_idx:06d}_{shape_str}_eta{eta}_rev{revs}.npz"
            filepath = subfolder / filename
            
            np.savez_compressed(filepath, images=images)
            logging.info(f"[IMGEN] Saved batch to {filepath}")
            
        elif save_format == 'png':
            # Handle 1-channel images by squeezing the last dimension
            if images.shape[-1] == 1:
                images = images[..., 0]
            # Handle 2-channel images by adding a zero channel
            elif images.shape[-1] == 2:
                zeros_like = np.zeros_like(images[..., 0:1])
                images = np.concatenate([images, zeros_like], axis=-1)
            elif images.shape[-1] >= 4:
                # >= 4 channels data, save each channel as grayscale image, concatenated such that [B, H, W, C] -> [B, H, W*C, 1]
                images = np.concatenate([images[..., i:i+1] for i in range(images.shape[-1])], axis=2)
                images = images[..., 0]
            # Save each image to appropriate subfolder
            for i, img in enumerate(images):
                global_idx = start_idx + i
                subfolder_idx = (global_idx // images_per_subfolder)*images_per_subfolder
                subfolder = Path(savedir) / f"sub_{subfolder_idx:06d}"
                subfolder.mkdir(parents=True, exist_ok=True)
                
                img = (img * 255).astype(np.uint8)
                filename = f"image_{global_idx:06d}.png"
                filepath = subfolder / filename
                # Use context managers to ensure file handles are closed promptly
                with Image.fromarray(img) as pil_img:
                    pil_img.save(filepath, format="PNG")
            
            logging.info(f"[IMGEN] Saved {len(images)} images (indices {start_idx}-{start_idx+len(images)-1})")

        else:
            raise ValueError(f"Unsupported save format: {save_format}. Use 'png', 'npz'.")

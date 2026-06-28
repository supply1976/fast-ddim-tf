import gc, sys
import logging
import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image
from pathlib import Path


class ImageGenerator:
    """
    Handles image generation and sampling for diffusion models.
    Separated from the main DiffusionModel for better modularity.
    """
    
    def __init__(self, diff_util, network, ema_network, timesteps, num_classes=None):
        """
        Initialize the ImageGenerator with a trained diffusion model.
        use ema_network for inference.
        
        Args:
            diff_util: Diffusion utility instance
            network: The main network for diffusion
            ema_network: The EMA (Exponential Moving Average) network for inference
            timesteps: Number of diffusion timesteps
            num_classes: Number of classes for conditional generation (optional)
        """
        self.diff_util = diff_util
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.num_classes = num_classes

    def _normalize_sampler(self, sampler):
        """Normalize sampler aliases used by configs and code."""
        sampler = (sampler or "ddim").lower()
        aliases = {
            "ddim": "ddim",
            "flow_euler": "flow_euler",
            "euler": "flow_euler",
            "ddim_euler": "flow_euler",
            "flow_heun": "flow_heun",
            "heun": "flow_heun",
            "ddim_heun": "flow_heun",
            "ddim_2nd": "flow_heun",
        }
        if sampler not in aliases:
            raise ValueError(
                "sampler must be one of: ddim, flow_euler, flow_heun"
            )
        sampler = aliases[sampler]
        if sampler != "ddim" and self.diff_util.ddim_eta != 0.0:
            raise ValueError(f"{sampler} is deterministic and requires DDIM_ETA: 0.0")
        return sampler
    
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
        # TODO
        """Denoise step for large canvas generation using patches.
        Args:
            x_t: Full canvas tensor at time t, tf.Tensor of shape (B, H, W, C)
            patch_size: Size of each patch
            stride: Stride between patches (controls overlap)
            t: Current timestep
            clip_denoise: Whether to clip denoising predictions
            labels: Optional class labels
        Returns:
            pred_noise_canvas, pred_image_canvas: Denoised full canvas tensor at time t, tf.Tensor of shape (B, H, W, C)
        """
        # split x_t into patches
        patches = tf.image.extract_patches(
            images=x_t,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        ) # (B, n_h, n_w, patch_size*patch_size*C)
        B = tf.shape(patches)[0]
        n_h = tf.shape(patches)[1]
        n_w = tf.shape(patches)[2]
        C = tf.shape(x_t)[-1]
        patches = tf.reshape(patches, (-1, patch_size, patch_size, C)) # (B*n_h*n_w, patch_size, patch_size, C)
        
        # Prepare timesteps for all patches
        tt = tf.fill((tf.shape(patches)[0],), t)
        inputs = [patches, tt]
        
        # Fix: Properly replicate labels for all patches
        # Each batch element's label should be repeated n_h*n_w times
        if labels is not None:
            labels_expanded = tf.repeat(labels, n_h * n_w)
            inputs.append(labels_expanded)
            
        y_pred = self.ema_network(inputs, training=False)
        pred_noise, pred_image = self.diff_util.get_pred_components(
            patches, tt, self.diff_util.pred_type, y_pred, clip_denoise=clip_denoise
        )
        # calculate the scores
        sigma_tt = tf.gather(self.diff_util.sigma_coefs, tt)[:, None, None, None]
        scores = - pred_noise / sigma_tt # (B*n_h*n_w, patch_size, patch_size, C)
        scores = tf.reshape(scores, (B, n_h, n_w, patch_size, patch_size, C))
        # blend scores (average) to reconstruct full canvas score
        # Initialize canvas and count for averaging
        canvas_shape = (B, (n_h - 1) * stride + patch_size, (n_w - 1) * stride + patch_size, C)
        score_canvas = tf.zeros(canvas_shape, dtype=scores.dtype)
        count = tf.zeros(canvas_shape, dtype=scores.dtype)
        
        # reconstruct full canvas from denoised patches, blending overlaps by averaging
        #pred_image_patches = tf.reshape(pred_image, (B, n_h, n_w, patch_size, patch_size, C))
        
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
        return pred_noise_canvas, pred_image_canvas
     
    @tf.function 
    def _denoise_step(self, x_t, t, clip_denoise, use_ema_model=True, labels=None):
        """Get predict noise and predict image (x0) at time t using input x_t."""
        tt = tf.fill((tf.shape(x_t)[0],), t)
        inputs = [x_t, tt]
        if labels is not None:
            inputs.append(labels)
        y_pred = None
        if use_ema_model:
            y_pred = self.ema_network(inputs, training=False)
        else:
            y_pred = self.network(inputs, training=False)
        pred_noise, pred_image = self.diff_util.get_pred_components(
            x_t, tt, self.diff_util.pred_type, y_pred, clip_denoise=clip_denoise
        )
        return pred_noise, pred_image

    def _denoise_for_task(
        self,
        samples,
        t,
        clip_denoise,
        labels,
        gen_task="random",
        canvas_patch_size=None,
        canvas_stride=None,
        use_ema_model=True,
    ):
        if gen_task == 'canvas_gen':
            return self._denoise_step_patches_to_canvas(
                samples,
                canvas_patch_size,
                canvas_stride,
                tf.constant(t, dtype=tf.int32),
                clip_denoise,
                labels,
            )
        return self._denoise_step(
            samples,
            tf.constant(t, dtype=tf.int32),
            clip_denoise,
            use_ema_model=use_ema_model,
            labels=labels,
        )

    def _sample_reverse_step(
        self,
        samples,
        t,
        s,
        eps_t,
        pred_image,
        sampler,
        clip_denoise,
        labels,
        gen_task="random",
        canvas_patch_size=None,
        canvas_stride=None,
        use_ema_model=True,
    ):
        tt = tf.cast(tf.fill((tf.shape(samples)[0],), t), tf.int32)
        ss = tf.cast(tf.fill((tf.shape(samples)[0],), s), tf.int32)
        if sampler == "ddim":
            return self.diff_util.p_sample_ddim(pred_image, eps_t, tt, ss)

        x_pred = self.diff_util.p_sample_flow(samples, eps_t, tt, ss, order=1)
        if sampler == "flow_euler" or int(s) == 0:
            return x_pred

        eps_s, _ = self._denoise_for_task(
            x_pred,
            int(s),
            clip_denoise,
            labels,
            gen_task=gen_task,
            canvas_patch_size=canvas_patch_size,
            canvas_stride=canvas_stride,
            use_ema_model=use_ema_model,
        )
        return self.diff_util.p_sample_flow(
            samples, eps_t, tt, ss, pred_noise_s=eps_s, order=2
        )
    
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
    
    def sample_images(self,
                      reverse_steps=100, 
                      num_images=20, 
                      clip_denoise=True,
                      use_ema_model=True, 
                      labels=None,
                      sampler="ddim",
                      ):
        """
        Generate samples and return them as numpy arrays.
        
        This is a lightweight variant used for inline evaluation where we only 
        need the final samples rather than saving them to disk.
        
        Args:
            reverse_steps: Steps for reverse diffusion
            num_images: Number of images to generate
            clip_denoise: If True, clip predicted x0 during reverse sampling
                and recompute the implied noise. Defaults to False.
            gen_inputs: Optional initial samples (if None, uses random noise)
            use_ema_model: Whether to use EMA network for inference
            labels: Optional class labels for conditional generation
            sampler: Reverse sampler. "ddim" keeps the original sampler,
                "flow_euler" is deterministic DDIM written as an ODE step,
                and "flow_heun" adds a 2nd-order endpoint correction.
            
        Returns:
            np.ndarray: Generated images as numpy array
        """
        sampler = self._normalize_sampler(sampler)
        # prepare initial samples and labels
        img_h, img_w, img_c = self.network.inputs[0].shape[1:]
        shape = (num_images, img_h, img_w, img_c)
        samples = tf.random.normal(shape=shape, dtype=tf.float32) 
        labels = self._prepare_labels(num_images, labels)
        reverse_timeindex, reverse_nextindex = self.diff_util.make_reverse_time_pairs(
            self.timesteps, reverse_steps
        )
         
        for t, s in tqdm.tqdm(zip(reverse_timeindex, reverse_nextindex), total=len(reverse_timeindex)):
            pred_noise, pred_image = self._denoise_step(
                samples, tf.constant(t, dtype=tf.int32), clip_denoise, use_ema_model, labels
            )
            samples = self._sample_reverse_step(
                samples,
                int(t),
                int(s),
                pred_noise,
                pred_image,
                sampler,
                clip_denoise,
                labels,
                use_ema_model=use_ema_model,
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
                                 sampler="ddim",
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
            sampler: Reverse sampler: "ddim", "flow_euler", or "flow_heun".
            savedir: Directory to save generated images
            save_intermediate: Whether to save denoised images at intermediate timesteps
            clip_denoise: If True, clip predicted x0 during reverse sampling
                and recompute the implied noise. Defaults to False.
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
        if sampler != "ddim" and self_guide_scale > 0.0:
            raise ValueError("SELF_GUIDE_SCALE is only supported with sampler='ddim'")
        if batch_size is None:
            batch_size = min(100, num_images)
        logging.info(f"[IMGEN] ===== Image Generation =====")
        logging.info(f"[IMGEN] Total images to generate: {num_images}")
        logging.info(f"[IMGEN] Batch size: {batch_size}")
        logging.info(f"[IMGEN] Number of batches: {(num_images + batch_size - 1) // batch_size}")
        logging.info(f"[IMGEN] Images per subfolder: {images_per_subfolder}")
        logging.info(f"[IMGEN] Reverse steps: {reverse_steps}")
        logging.info(f"[IMGEN] Sampler: {sampler}")
        
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
        img_h, img_w, img_c = self.network.inputs[0].shape[1:]
        
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
        t_start = self.timesteps
        if gen_task == 'img2img':
            assert base_images is not None
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
            t_start, reverse_steps
        )
        eps_prev = None
        
        logging.debug(f"[IMGEN] Starting reverse diffusion with {len(reverse_timeindex)} steps")
        
        progress_bar = tqdm.tqdm(
            list(zip(reverse_timeindex, reverse_nextindex)),
            desc=f"Batch generation",
            total=len(reverse_timeindex),
            file=sys.stderr,
            ncols=100,
            disable=None  # Auto-disable if not a TTY
            )
        
        for t, s in progress_bar:
            # main loop for reverse diffusion
            # ex: 
            # t = [1000, 990, 980, ..., 10] for stride=10, steps=100
            # s = [990, 980, ..., 0]
            tt = tf.cast(tf.fill((tf.shape(samples)[0],), t), tf.int32) # [B,]
            
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
                    base_images_t, _ = self.diff_util.q_sample(base_images, tt, noise0)
                    samples = samples * inpaint_mask + base_images_t * (1 - inpaint_mask)
                else:
                    samples = samples * inpaint_mask + base_images * (1 - inpaint_mask)
            
            eps_t, pred_image = self._denoise_for_task(
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
                eps_guide = eps_t + self_guide_scale * (eps_prev - eps_t) if eps_prev is not None else eps_t
                ss = tf.cast(tf.fill((tf.shape(samples)[0],), s), tf.int32) # [B,]
                samples = self.diff_util.p_sample_ddim(pred_image, eps_guide, tt, ss)
                eps_prev = eps_t
            else:
                samples = self._sample_reverse_step(
                    samples,
                    int(t),
                    int(s),
                    eps_t,
                    pred_image,
                    sampler,
                    clip_denoise,
                    batch_labels,
                    gen_task=gen_task,
                    canvas_patch_size=canvas_patch_size,
                    canvas_stride=canvas_stride,
                    use_ema_model=True,
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

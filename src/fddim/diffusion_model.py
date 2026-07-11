import os
import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from .image_generator import ImageGenerator


class DiffusionModel(keras.Model):
    """Diffusion Model with configurable loss weighting strategies.
    
    Loss Weighting Options:
    -----------------------
    1. 'constant': Standard uniform weighting across all timesteps (default behavior)
       - Each timestep contributes equally to the loss
       - Simple and straightforward, suitable for most use cases
    
    2. 'min_snr': Min-SNR weighting strategy
       - Addresses the issue of imbalanced learning across timesteps
       - Applies adaptive weighting based on Signal-to-Noise Ratio (SNR)
       - Weight formula depends on prediction type:
         noise: min(SNR(t), gamma) / SNR(t)
         image: min(SNR(t), gamma)
         velocity: min(SNR(t), gamma) / (SNR(t) + 1)
       - Where SNR(t) = alpha_t / (1 - alpha_t)
       - Reference: "Efficient Diffusion Training via Min-SNR Weighting Strategy" (Hang et al., 2023)
       - Benefits: Better sample quality and faster convergence
       - Recommended gamma: 5.0 (default), can be tuned between 1-10
    
    Args:
        network: Main U-Net model
        ema_network: Exponential Moving Average network for generation
        diff_util: DiffusionUtility instance containing noise scheduling
        num_classes: Number of classes for conditional generation (None for unconditional)
        save_period: Save model every N epochs
        ema: EMA decay rate (default 0.995)
        loss_weight_type: Loss weighting strategy - 'constant' or 'min_snr'
        min_snr_gamma: Gamma parameter for min-SNR weighting (default 5.0)
        gradient_accumulation_steps: Number of steps to accumulate gradients before applying an update
    """
    def __init__(self, network, ema_network, diff_util, num_classes, save_period=100, ema=0.995,
                 loss_weight_type='constant', min_snr_gamma=5.0, gradient_accumulation_steps=1):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.diff_util = diff_util
        self.timesteps = diff_util.timesteps
        self.ema = ema
        self.clip_denoise = diff_util.clip_denoise
        self.num_classes = num_classes
        self.save_period = save_period
        self.loss_weight_type = loss_weight_type
        self.min_snr_gamma = min_snr_gamma
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        assert self.diff_util.pred_type in ['noise', 'image', 'velocity'], \
            "pred_type must be one of [noise, image, velocity]"
        assert self.loss_weight_type in ['constant', 'min_snr'], \
            "loss_weight_type must be one of ['constant', 'min_snr']"
        # Precompute SNR values for min-SNR weighting
        self._precompute_snr_values()
        # Gradient accumulation state
        self._use_grad_accum = self.gradient_accumulation_steps > 1
        if self._use_grad_accum:
            self._grad_accum_steps = tf.constant(self.gradient_accumulation_steps, dtype=tf.int64)
            self._grad_accum_counter = tf.Variable(
                0, dtype=tf.int64, trainable=False, name="grad_accum_counter"
            )
            self._grad_accum_vars = [
                tf.Variable(tf.zeros_like(weight), trainable=False)
                for weight in self.network.trainable_weights
            ]
        # Initialize generation components
        self.image_generator = ImageGenerator(diff_util, network, ema_network, self.timesteps, num_classes)

    def _precompute_snr_values(self):
        """Precompute SNR values for all timesteps.
        SNR(t) = alpha_t / (1 - alpha_t) = alpha_t / sigma_t^2
        """
        if self.loss_weight_type == 'min_snr':
            # Get alphas from diff_util (numpy array of shape (timesteps+1,))
            alphas = self.diff_util.alphas
            # Compute SNR for each timestep
            snr = alphas / (1.0 - alphas + 1e-8)  # Add epsilon for numerical stability
            # Convert to TensorFlow constant for efficient lookup
            self.snr_values = tf.constant(snr, dtype=tf.float32)
        else:
            self.snr_values = None
    
    def _compute_loss_weights(self, t):
        """Compute loss weights based on timestep and weighting strategy.
        
        Args:
            t: Timestep tensor of shape (batch_size,)
            
        Returns:
            Loss weights of shape (batch_size, 1, 1)
        """
        if self.loss_weight_type == 'constant':
            # Constant weighting (current behavior)
            return tf.ones_like(t, dtype=tf.float32)[:, None, None]
        elif self.loss_weight_type == 'min_snr':
            snr_t = tf.gather(self.snr_values, t)
            min_snr = tf.minimum(snr_t, self.min_snr_gamma)

            # Min-SNR weights depend on the prediction parameterization.
            # See Hang et al., "Efficient Diffusion Training via Min-SNR
            # Weighting Strategy" (arXiv:2303.09556), Sec. 4.2.
            if self.diff_util.pred_type == 'noise':
                weights = tf.math.divide_no_nan(min_snr, snr_t)
            elif self.diff_util.pred_type == 'image':
                weights = min_snr
            elif self.diff_util.pred_type == 'velocity':
                weights = min_snr / (snr_t + 1.0)
            else:
                raise ValueError(f"Unknown pred_type: {self.diff_util.pred_type}")
            return weights[:, None, None]
        else:
            raise ValueError(f"Unknown loss_weight_type: {self.loss_weight_type}")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.noise_loss_tracker,
            self.image_loss_tracker,
        ]

    def _apply_gradients(self, gradients):
        grads_and_vars = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.network.trainable_weights)
            if gradient is not None
        ]
        if not grads_and_vars:
            raise ValueError("No gradients were produced for the trainable weights")

        valid_grads, valid_vars = zip(*grads_and_vars)
        clipped_grads, _ = tf.clip_by_global_norm(valid_grads, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(clipped_grads, valid_vars))
        self._update_ema_weights()

    def _apply_accumulated_gradients(self):
        scaled_grads = [
            acc / tf.cast(self._grad_accum_steps, acc.dtype)
            for acc in self._grad_accum_vars
        ]
        self._apply_gradients(scaled_grads)
        for acc in self._grad_accum_vars:
            acc.assign(tf.zeros_like(acc))
        self._grad_accum_counter.assign(0)
        return tf.constant(0)

    def _compute_training_losses(self, images, noises, images_t, t, v_t, y_pred):
        pred_noise, pred_image = self.diff_util.get_pred_components(
            images_t, t, self.diff_util.pred_type, y_pred, clip_denoise=False,
        )
        if self.diff_util.pred_type == 'noise':
            loss = self.loss(noises, y_pred)
        elif self.diff_util.pred_type == 'image':
            loss = self.loss(images, y_pred)
        elif self.diff_util.pred_type == 'velocity':
            loss = self.loss(v_t, y_pred)
        else:
            raise ValueError("pred_type must be one of [noise, image, velocity]")
        noise_loss = self.loss(noises, pred_noise)
        image_loss = self.loss(images, pred_image)
        return loss, noise_loss, image_loss

    @tf.function
    def train_step(self, data):
        if isinstance(data, (list, tuple)):
            images, labels = data
        else:
            images, labels = data, None
        batch_size = tf.shape(images)[0]
        t = tf.random.uniform(
            minval=1, maxval=self.timesteps + 1, shape=(batch_size,), dtype=tf.int32)
        with tf.GradientTape() as tape:
            noises = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
            images_t, v_t = self.diff_util.q_sample(images, t, noises)
            inputs = [images_t, t]
            if labels is not None:
                # randomly null the label with a small probability (Classifier-free guide)
                null_mask = tf.random.uniform([]) <= 0.1
                labels = tf.cond(null_mask, lambda: tf.zeros_like(labels), lambda: labels)
                inputs.append(labels)
            y_pred = self.network(inputs, training=True)
            loss, noise_loss, image_loss = self._compute_training_losses(
                images, noises, images_t, t, v_t, y_pred
            )
            
            # Apply loss weights based on timestep
            loss_weights = self._compute_loss_weights(t)
            # Reduce spatially and apply weights
            # Loss shape is (batch, height, width) after loss function, so reduce over spatial dims
            loss = tf.reduce_mean(loss, axis=[1, 2], keepdims=True)
            weighted_loss = loss * loss_weights
            loss = tf.reduce_mean(weighted_loss)

        gradients = tape.gradient(loss, self.network.trainable_weights)
        if not self._use_grad_accum:
            self._apply_gradients(gradients)
        else:
            for acc, grad in zip(self._grad_accum_vars, gradients):
                if grad is not None:
                    acc.assign_add(grad)
            self._grad_accum_counter.assign_add(1)
            tf.cond(
                tf.equal(self._grad_accum_counter, self._grad_accum_steps),
                self._apply_accumulated_gradients,
                lambda: tf.constant(0),
            )

        # Reduce noise and image losses to scalars for tracking
        noise_loss_scalar = tf.reduce_mean(noise_loss)
        image_loss_scalar = tf.reduce_mean(image_loss)

        self.loss_tracker.update_state(loss)
        self.noise_loss_tracker.update_state(noise_loss_scalar)
        self.image_loss_tracker.update_state(image_loss_scalar)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def _update_ema_weights(self):
        for ema_weight, weight in zip(self.ema_network.trainable_weights, self.network.trainable_weights):
            ema_weight.assign(ema_weight * self.ema + (1 - self.ema) * weight)

    @tf.function
    def test_step(self, data):
        if isinstance(data, (list, tuple)):
            images, labels = data
        else:
            images, labels = data, None
        batch_size = tf.shape(images)[0]
        t = tf.random.uniform(minval=1, maxval=self.timesteps + 1, shape=(batch_size,), dtype=tf.int32)
        noises = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        images_t, v_t = self.diff_util.q_sample(images, t, noises)
        inputs = [images_t, t]
        if labels is not None:
            inputs.append(labels)
        y_pred = self.ema_network(inputs, training=False)
        loss, noise_loss, image_loss = self._compute_training_losses(
            images, noises, images_t, t, v_t, y_pred
        )
        
        # Reduce all losses to scalars for tracking
        loss_scalar = tf.reduce_mean(loss)
        noise_loss_scalar = tf.reduce_mean(noise_loss)
        image_loss_scalar = tf.reduce_mean(image_loss)
        
        self.loss_tracker.update_state(loss_scalar)
        self.noise_loss_tracker.update_state(noise_loss_scalar)
        self.image_loss_tracker.update_state(image_loss_scalar)
        return {m.name: m.result() for m in self.metrics}

    def save_models(self, epoch, logs='mylog.txt', savedir=None):
        # save EMA model on epoch end
        if savedir is None:
            savedir = './saved_models'
        os.makedirs(savedir, exist_ok=True)
        epo = str(epoch+1).zfill(5)
        output_name = "unet_tf" + tf.__version__
        if self.save_period is not None:
            if (epoch+1) % self.save_period == 0 and epoch > 0:
                path_unet_epo = os.path.join(savedir, output_name+f"epoch_{epo}")
                self.ema_network.save(path_unet_epo+"_ema" + ".keras", include_optimizer=False)
    
    # Convenience methods that delegate to the image generator
    def sample_images(self, **kwargs):
        """Generate samples using the ImageGenerator."""
        return self.image_generator.sample_images(**kwargs) # numpy array
    
    def generate_images_and_save(self, **kwargs):
        """Generate and save images using ImageGenerator."""
        # Let ImageGenerator handle its own memory logging
        output_dict = self.image_generator.generate_images_and_save(**kwargs)
        return output_dict

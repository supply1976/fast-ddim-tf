"""Keras callbacks used for training the diffusion model."""

import os
import logging
import time
import csv
import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow import keras # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input # type: ignore
from tqdm import tqdm

from .metrics import (
    GZIP_RAW_METRIC_FIELDS,
    PAIRWISE_METRIC_FIELDS,
    compute_pairwise_image_metrics,
    format_metric_values,
    gzip_raw_metrics,
)


class RobustCSVLogger(keras.callbacks.Callback):
    """CSV logger with error handling for network filesystem issues.
    
    This callback handles stale file handles and other I/O errors that can occur
    when writing to network filesystems (NFS). It will retry writes and log errors
    without crashing the training process.
    """
    
    def __init__(self, filename, separator=',', append=False, max_retries=3, period=None):
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        self.max_retries = max_retries
        self.period = period
        self.keys = None
        self.file_handle = None
        self.writer = None
        self._open_file()
    
    def _open_file(self):
        """Open the CSV file with error handling."""
        try:
            if self.append and os.path.exists(self.filename):
                self.file_handle = open(self.filename, 'a', newline='')
                # Read existing headers
                with open(self.filename, 'r') as f:
                    reader = csv.reader(f)
                    try:
                        self.keys = next(reader)
                    except StopIteration:
                        self.keys = None
            else:
                self.file_handle = open(self.filename, 'w', newline='')
            
            self.writer = csv.writer(self.file_handle, delimiter=self.separator)
        except Exception as e:
            logging.error(f"[CSVLogger] Failed to open {self.filename}: {e}")
            self.file_handle = None
            self.writer = None
    
    def _write_row(self, row_dict):
        """Write a row with retry logic for I/O errors."""
        if self.writer is None:
            return False
        
        for attempt in range(self.max_retries):
            try:
                if self.keys is None:
                    self.keys = ["epoch"] + sorted(k for k in row_dict.keys() if k != "epoch")
                    self.writer.writerow(self.keys)
                
                row = [row_dict.get(key, '') for key in self.keys]
                self.writer.writerow(row)
                self.file_handle.flush()
                return True
            except (OSError, IOError) as e:
                logging.warning(f"[CSVLogger] Write attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)  # Brief pause before retry
                    # Try to reopen file
                    try:
                        if self.file_handle:
                            self.file_handle.close()
                    except:
                        pass
                    self._open_file()
                else:
                    logging.error(f"[CSVLogger] Failed to write to {self.filename} after {self.max_retries} attempts")
                    return False
            except Exception as e:
                logging.error(f"[CSVLogger] Unexpected error writing to {self.filename}: {e}")
                return False
        
        return False
    
    def on_epoch_end(self, epoch, logs=None):
        """Write epoch metrics to CSV."""
        if logs is None:
            return
        if self.period is not None and (epoch + 1) % self.period != 0:
            return
        
        row_dict = {'epoch': epoch + 1}
        row_dict.update(logs)
        self._write_row(row_dict)
    
    def on_train_end(self, logs=None):
        """Close the file handle."""
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception as e:
                logging.warning(f"[CSVLogger] Error closing file: {e}")


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup and cosine decay."""

    def __init__(self, base_lr, total_steps, warmup_steps, min_lr=0.0):
        super().__init__()
        if total_steps is None or int(total_steps) <= 0:
            raise ValueError("total_steps must be a positive integer")
        if warmup_steps is None:
            raise ValueError("warmup_steps must be provided")
        if int(warmup_steps) <= 0:
            raise ValueError("warmup_steps must be > 0")
        if int(warmup_steps) >= int(total_steps):
            raise ValueError("warmup_steps must be < total_steps")
        self.base_lr = base_lr
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = min_lr

    def __call__(self, step):
        warmup_lr = self.base_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        cosine_steps = tf.cast(step - self.warmup_steps, tf.float32)
        cosine_total = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * cosine_steps / cosine_total))
        decayed = (self.base_lr - self.min_lr) * cosine_decay + self.min_lr
        return tf.where(step < self.warmup_steps, warmup_lr, decayed)


class TQDMProgressBar(keras.callbacks.Callback):
    """tqdm based progress bar for model.fit."""

    def __init__(self):
        super().__init__()
        self.epochs = None
        self.steps_per_epoch = None
        self.current_epoch = None
        self._epoch_start_time = None

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps_per_epoch = self.params.get('steps')

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self._epoch_start_time = time.time()
        self.pbar = tqdm(total=self.steps_per_epoch,
                         desc=f"Epoch {epoch+1}/{self.epochs}",
                         leave=False)
        logging.info(f"[TRAIN] Epoch {epoch+1}/{self.epochs} started")

    def on_train_batch_end(self, batch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        gs = int(self.model.optimizer.iterations.numpy())
        
        # Prepare postfix with lr and global step
        postfix = {'lr': f"{lr:.4e}", 'gs': gs}
        
        # Add loss information if available
        if logs:
            if 'loss' in logs:
                postfix['loss'] = f"{logs['loss']:.4f}"
            # Add validation loss if available (during validation)
            if 'val_loss' in logs:
                postfix['val_loss'] = f"{logs['val_loss']:.4f}"
        
        self.pbar.set_postfix(postfix)
        self.pbar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        # Display epoch summary with final metrics
        logs = logs or {}
        epoch_summary = f"Epoch {epoch+1}/{self.epochs} - "
        metrics = []
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        metrics.append(f"lr: {lr:.4e}")
        if 'loss' in logs:
            metrics.append(f"loss: {logs['loss']:.4f}")
        if 'n_loss' in logs:
            metrics.append(f"n_loss: {logs['n_loss']:.4f}")
        if 'i_loss' in logs:
            metrics.append(f"i_loss: {logs['i_loss']:.4f}")
        if 'v_loss' in logs:
            metrics.append(f"v_loss: {logs['v_loss']:.4f}")
        if 'val_loss' in logs:
            metrics.append(f"val_loss: {logs['val_loss']:.4f}")
        if 'val_n_loss' in logs:
            metrics.append(f"val_n_loss: {logs['val_n_loss']:.4f}")
        if 'val_i_loss' in logs:
            metrics.append(f"val_i_loss: {logs['val_i_loss']:.4f}")
        if 'val_v_loss' in logs:
            metrics.append(f"val_v_loss: {logs['val_v_loss']:.4f}")
        if metrics:
            epoch_summary += " - ".join(metrics)
            elapsed = None
            if self._epoch_start_time is not None:
                elapsed = time.time() - self._epoch_start_time
            gs = int(self.model.optimizer.iterations.numpy())
            if elapsed is not None:
                logging.info(
                    f"[TRAIN] {epoch_summary} - time: {elapsed:.2f}s - gs: {gs}"
                )
            else:
                logging.info(f"[TRAIN] {epoch_summary} - gs: {gs}")
        
        self.pbar.close()


class BestModelCheckpoint(keras.callbacks.Callback):
    """Save both the best EMA model and non-EMA model weights based on a monitored metric.
    
    This callback monitors a specified metric (e.g., 'loss', 'val_loss') and saves
    both the EMA and non-EMA model weights when the metric improves. 
    
    Args:
        filepath: Path where the model weights should be saved.
        monitor: Metric name to monitor (e.g., 'loss', 'val_loss').
        mode: One of {'min', 'max'}. In 'min' mode, saves when monitored metric decreases;
              in 'max' mode, saves when it increases.
        verbose: Verbosity level (0 or 1).
    """
    
    def __init__(self, filepath, monitor='loss', mode='min', verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got '{mode}'")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose > 0:
                logging.warning(
                    f"[CALLBACK] EMAModelCheckpoint: {self.monitor} not available in logs, skipping."
                )
            return
        
        if self.monitor_op(current, self.best):
            if self.verbose > 0:
                logging.info(
                    f"[CALLBACK] Epoch {epoch+1}: best {self.monitor} improved; saving full models to Keras"
                )
            self.best = current
            # Save both the EMA and non-EMA full models (architecture + weights) in Keras format.
            fp = str(self.filepath)
            #network_path = fp + '.h5', legacy for keras < 3.0, for keras 3.0+, the recommended extension is .keras
            network_path = fp + '.keras'
            #ema_path = fp + '_ema.h5', legacy for keras < 3.0, for keras 3.0+, the recommended extension is .keras
            ema_path = fp + '_ema.keras'
            # Save models without optimizer state to keep files lightweight and portable
            try:
                self.model.network.save(network_path, include_optimizer=False)
                self.model.ema_network.save(ema_path, include_optimizer=False)
                if self.verbose > 0:
                    logging.info(f"[CALLBACK] Saved models: {network_path} and {ema_path}")
            except Exception:
                logging.exception("[CALLBACK] Error saving full models to Keras")


class InlineImageGenerationCallback(keras.callbacks.Callback):
    """Keras callback to generate and save images at the end of every N epochs."""
    
    def __init__(
        self,
        reverse_steps=50,
        period=10,
        savedir='./inline_gen',
        num_images=4,
        labels=None,
        ssim_pairs=1000,
        duplicate_ssim_threshold=0.98,
        duplicate_l2_threshold=0.02,
        metrics_csv_path=None,
    ):
        super().__init__()
        self.reverse_steps = reverse_steps
        self.period = period
        self.savedir = savedir
        self.num_images = num_images
        self.labels = labels
        self.ssim_pairs = ssim_pairs
        self.duplicate_ssim_threshold = duplicate_ssim_threshold
        self.duplicate_l2_threshold = duplicate_l2_threshold
        self.metrics_csv_path = metrics_csv_path
        os.makedirs(self.savedir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.period == 0:
            logging.info(
                f"[CALLBACK] Inline generating {self.num_images} images at epoch {epoch+1}..."
            )
            savedir = os.path.join(self.savedir, f"epoch_{str(epoch+1).zfill(5)}")
            os.makedirs(savedir, exist_ok=True)
            images = None
            # generate EMA model images for inline check
            try:
                images = self.model.sample_images(
                    reverse_steps=self.reverse_steps,
                    num_images=self.num_images,
                    use_ema_model=True,
                    labels=self.labels,
                )
            except Exception:
                logging.exception("[CALLBACK] Inline image generation failed")
            if images is not None:
                metric_images = images
                gzip_metrics = gzip_raw_metrics(metric_images)
                pairwise_metrics = compute_pairwise_image_metrics(
                    metric_images,
                    num_pairs=self.ssim_pairs,
                    pair_seed=epoch + 1,
                    duplicate_ssim_threshold=self.duplicate_ssim_threshold,
                    duplicate_l2_threshold=self.duplicate_l2_threshold,
                )
                metrics_row = {
                    "epoch": epoch + 1,
                    "num_images": int(metric_images.shape[0]),
                    "image_height": int(metric_images.shape[1]),
                    "image_width": int(metric_images.shape[2]),
                    "image_channels": int(metric_images.shape[3]),
                    "num_pairs": 0,
                    "reverse_steps": int(self.reverse_steps),
                    "savedir": savedir,
                    **gzip_metrics,
                    **pairwise_metrics,
                }
                if metrics_row["num_pairs"] > 0:
                    logging.info(
                        "[CALLBACK] Inline pairwise SSIM: "
                        f"{metrics_row['pairwise_ssim']:.6f} "
                        f"+/- {metrics_row['pairwise_ssim_std']:.6f} "
                        f"max={metrics_row['pairwise_ssim_max']:.6f}, "
                    )
                self._write_metrics_row(metrics_row)
                logging.info(
                    "[CALLBACK] Inline gzip raw compression: "
                    f"ratio={metrics_row['gzip_raw_compression_ratio']:.6f}, "
                    f"bits/value={metrics_row['gzip_raw_bits_per_value']:.6f}"
                )
                if images.shape[-1]==2:
                    # 2 channels, add a zero channel for saving
                    zeros = np.zeros_like(images)
                    images = np.concatenate([images, zeros[..., 0:1]], axis=-1)
                elif images.shape[-1]>=4:
                    # for >= 4 channels, concatenate such that [B, H, W, C] -> [B, H, W*C, 1]
                    images = np.concatenate([images[..., i:i+1] for i in range(images.shape[-1])], axis=2)
                # save images to png files
                for i, img in enumerate(images):
                    img = (img * 255.0).astype(np.uint8)
                    filename = f"img_{str(i+1).zfill(4)}.png"
                    filepath = os.path.join(savedir, filename)
                    keras.utils.save_img(filepath, img)
                logging.info(f"[CALLBACK] Inline Images Gen saved to {savedir}")

    def _write_metrics_row(self, row):
        if self.metrics_csv_path is None:
            return
        os.makedirs(os.path.dirname(self.metrics_csv_path), exist_ok=True)
        fieldnames = [
            "epoch",
            "num_images",
            "image_height",
            "image_width",
            "image_channels",
            "num_pairs",
            "reverse_steps",
            *PAIRWISE_METRIC_FIELDS,
            "near_duplicate_ssim_threshold",
            "near_duplicate_l2_threshold",
            "near_duplicate_rate_ssim",
            "near_duplicate_rate_l2",
            "near_duplicate_rate_any",
            *GZIP_RAW_METRIC_FIELDS,
            "savedir",
        ]
        write_header = not os.path.exists(self.metrics_csv_path)
        try:
            with open(self.metrics_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                csv_row = format_metric_values(
                    {fieldname: row.get(fieldname) for fieldname in fieldnames},
                    as_strings=True,
                )
                writer.writerow(csv_row)
        except Exception:
            logging.exception(f"[CALLBACK] Failed to write inline metrics CSV: {self.metrics_csv_path}")


# TODO (not completed): Implement a more comprehensive callback for FID evaluation)
class InlineEvalCallback(keras.callbacks.Callback):
    """Generate samples during training and compute FID."""

    def __init__(self, valid_ds, eval_interval=1000, savedir=None, patience=3, num_images=16):
        super().__init__()
        self.valid_ds = valid_ds
        self.eval_interval = eval_interval
        self.savedir = savedir
        self.patience = patience
        self.num_images = num_images
        self.best = np.inf
        self.wait = 0
        self.valid_iter = iter(valid_ds)
        self.inception = InceptionV3(include_top=False, pooling='avg',
                                     weights="./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                     input_shape=(299, 299, 3))

    def _calc_fid(self, real, fake):
        real = tf.image.resize(real, (299, 299))
        fake = tf.image.resize(fake, (299, 299))
        real = preprocess_input(real * 255.0)
        fake = preprocess_input(fake * 255.0)
        act1 = self.inception(real, training=False)
        act2 = self.inception(fake, training=False)
        mu1 = tf.reduce_mean(act1, axis=0)
        mu2 = tf.reduce_mean(act2, axis=0)
        x1 = act1 - mu1
        x2 = act2 - mu2
        sigma1 = tf.matmul(x1, x1, transpose_a=True) / tf.cast(tf.shape(act1)[0]-1, tf.float32)
        sigma2 = tf.matmul(x2, x2, transpose_a=True) / tf.cast(tf.shape(act2)[0]-1, tf.float32)
        diff = mu1 - mu2
        s12 = tf.matmul(sigma1, sigma2)
        covmean = sqrtm(s12.numpy())
        covmean = tf.cast(tf.math.real(covmean), tf.float32)
        fid = tf.tensordot(diff, diff, axes=1) + tf.linalg.trace(
            sigma1 + sigma2 - 2.0 * covmean)
        return float(fid.numpy())

    def on_train_batch_end(self, batch, logs=None):
        step = int(self.model.optimizer.iterations.numpy())
        if step == 0 or step % self.eval_interval != 0:
            return
        try:
            real_images = next(self.valid_iter)
            if isinstance(real_images, (list, tuple)):
                real_images, labels_batch = real_images
            else:
                labels_batch = None
            if isinstance(real_images, dict):
                real_images = real_images['image']
        except StopIteration:
            self.valid_iter = iter(self.valid_ds)
            real_images = next(self.valid_iter)
            if isinstance(real_images, (list, tuple)):
                real_images, labels_batch = real_images
            else:
                labels_batch = None
            if isinstance(real_images, dict):
                real_images = real_images['image']
        real_images = (real_images + 1.0) / 2.0
        fake_images = self.model.sample_images(num_images=self.num_images, labels=labels_batch)
        fid_value = self._calc_fid(real_images[:self.num_images], fake_images[:self.num_images])
        logging.info(f"[EVAL] step {step}, FID={fid_value:.6f}")
        if fid_value < self.best:
            self.best = fid_value
            self.wait = 0
            if self.savedir is not None:
                best_path = os.path.join(self.savedir, "best_model.keras")
                self.model.ema_network.save(best_path, include_optimizer=False)
                logging.info(f"[EVAL] best model saved to {best_path}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logging.info("[EVAL] Early stopping triggered")
                self.model.stop_training = True

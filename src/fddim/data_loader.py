import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import numpy as np
from functools import partial
import tensorflow as tf


class DataLoader:
    def __init__(
        self,
        data_dir,
        img_resize=None,
        crop_size=None,
        crop_type='center',
        crop_position=None,
        image_key='image',
        label_key=None,
        file_format='auto',
        augment=False,
        augment_type=None,
        validation_split=None,
        cache=True,
    ):
        """
        Enhanced DataLoader for DDPM v3 with flexible cropping, normalization, and multi-format support.

        This loader:
          - Recursively searches for .npz, .jpg/.jpeg, or .png files under data_dir.
          - Enforces single file format - raises error if multiple formats found.
          - Loads arrays (H, W, C) by image_key (and optionally label_key) for .npz files.
          - Loads JPEG/PNG images directly using TensorFlow's optimized decoders.
          - All preprocessing (resize, crop, augment) uses TensorFlow ops for optimal performance.
          - Resize input array (H, W, C) first to have the smaller dimension equal to img_resize if provided, 
          - Keep aspect ratio, then center crop to (img_size, img_size, C).
          - Applies optional cropping strategies (center, random, corner).
          - Applies optional augmentation (e.g., flips, 90-degree rotation).
          - Normalizes images from [0, 1] to [-1, 1].
          - Builds tf.data.Dataset objects for training and validation, with sensible caching,
            shuffling, repeating, and prefetch settings.

        Args:
            data_dir (str): Directory containing input files (.npz or .jpg/.jpeg).
            img_resize (int): 
                - if None, use original image size.
                - if provided, resize input array first to have smaller dimension equal to img_resize,
                  keep aspect ratio, then center crop to (img_resize, img_resize).
            crop_size (int, optional): Size used during cropping. If None, cropping is skipped.
            crop_type (str): Cropping strategy, one of:
                - 'center': Center crop (deterministic)
                - 'random': Random crop (per-example randomness when is_training=True)
                - 'corner': Crop from a specific corner (requires crop_position)
            crop_position (str, optional): Corner position for 'corner' crop_type:
                'top_left', 'top_right', 'bottom_left', 'bottom_right'.
            image_key (str): Key name for the image array in each .npz file (ignored for JPEG).
            label_key (str, optional): Key name for the label in each .npz file (ignored for JPEG). If None, no labels are loaded.
            file_format (str): File format to search for:
                - 'auto': Auto-detect based on files found (default, enforces single format)
                - '.npz': NumPy compressed arrays
                - '.jpg' or '.jpeg': JPEG images
                - '.png': PNG images
            augment (bool): Whether to apply augmentation in training mode.
            augment_type (str, optional): Augmentation type. Supported:
                - 'fliplr': Random horizontal flip
                - 'flipud': Random vertical flip
                - 'rotate': Random 90-degree rotation (counter-clockwise)
                - 'flip-rotate': Combined random flip and rotation
                - 'center_defect': Experimental small zero-mask at image center
            validation_split (float, optional): Fraction of files used for validation. If None, all files are training.
            cache (bool): If True, cache the dataset in memory when deterministic.

        Notes:
            - Dataset directory must contain only ONE file format (enforced strictly).
            - All resize, crop, and augmentation operations use TensorFlow for GPU acceleration.
            - Resize first if img_resize is provided.
            - Cropping (if enabled) happens before augmentation.
            - Augmentation happens only when is_training=True.
            - JPEG/PNG images are loaded with TensorFlow's optimized decoders for best performance.

        Returns:
            None. Use _get_dataset() to obtain (train_ds, valid_ds).
        """
        self.CLIP_MAX = 1.0
        self.CLIP_MIN = -1.0
        self.img_resize = img_resize
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.crop_position = crop_position
        self.augment = augment
        self.augment_type = augment_type
        self.data_dir = os.path.abspath(data_dir)
        self.image_key = image_key
        self.label_key = label_key
        self.validation_split = validation_split
        self.cache = cache
        self.img_shape = None

        # Validate crop parameters
        valid_crop_types = [None, 'center', 'random', 'corner']
        assert self.crop_type in valid_crop_types, f"crop_type must be one of {valid_crop_types}"

        if self.crop_type == 'corner':
            valid_positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
            assert self.crop_position in valid_positions, (
                f"crop_position must be one of {valid_positions} for corner cropping"
            )

        assert os.path.exists(self.data_dir), f"data_dir {self.data_dir} not exists"
        assert os.path.isdir(self.data_dir), f"data_dir {self.data_dir} is not a directory"
        assert file_format is not None, "file_format should not be None"

        # Discover data files with strict single-format validation
        self.all_datafiles = []
        self.data_format = None  # Will be set to 'npz', 'jpg', or 'png'
        
        if file_format == 'auto':
            # Auto-detect file format with strict validation
            npz_files = []
            jpg_files = []
            png_files = []
            
            for root, dirs, files in os.walk(self.data_dir):
                for fn in files:
                    file_path = os.path.join(root, fn)
                    if fn.endswith('.npz'):
                        npz_files.append(file_path)
                    elif fn.lower().endswith(('.jpg', '.jpeg')):
                        jpg_files.append(file_path)
                    elif fn.lower().endswith('.png'):
                        png_files.append(file_path)
            
            # Count formats found
            formats_found = []
            if len(npz_files) > 0:
                formats_found.append(f".npz ({len(npz_files)} files)")
            if len(jpg_files) > 0:
                formats_found.append(f".jpg/.jpeg ({len(jpg_files)} files)")
            if len(png_files) > 0:
                formats_found.append(f".png ({len(png_files)} files)")
            
            # Enforce single format rule
            if len(formats_found) == 0:
                raise ValueError(
                    f"No supported files found in {self.data_dir}.\n"
                    "Supported formats: .npz, .jpg, .jpeg, .png"
                )
            elif len(formats_found) > 1:
                raise ValueError(
                    f"Multiple file formats detected in {self.data_dir}:\n"
                    f"  Found: {', '.join(formats_found)}\n"
                    "Dataset directory must contain only ONE file format.\n"
                    "Please organize your dataset to use a single format (.npz, .jpg/.jpeg, or .png)."
                )
            
            # Set format and files
            if len(npz_files) > 0:
                self.data_format = 'npz'
                self.all_datafiles = npz_files
            elif len(jpg_files) > 0:
                self.data_format = 'jpg'
                self.all_datafiles = jpg_files
            else:  # len(png_files) > 0
                self.data_format = 'png'
                self.all_datafiles = png_files
        else:
            # Use specified file format
            if file_format == '.npz':
                self.data_format = 'npz'
            elif file_format in ['.jpg', '.jpeg']:
                self.data_format = 'jpg'
            elif file_format == '.png':
                self.data_format = 'png'
            else:
                raise ValueError(
                    f"Unsupported file_format: {file_format}.\n"
                    "Use 'auto', '.npz', '.jpg', '.jpeg', or '.png'"
                )
            
            for root, dirs, files in os.walk(self.data_dir):
                for fn in files:
                    file_path = os.path.join(root, fn)
                    if file_format == '.npz' and fn.endswith('.npz'):
                        self.all_datafiles.append(file_path)
                    elif file_format in ['.jpg', '.jpeg'] and fn.lower().endswith(('.jpg', '.jpeg')):
                        self.all_datafiles.append(file_path)
                    elif file_format == '.png' and fn.lower().endswith('.png'):
                        self.all_datafiles.append(file_path)
        
        assert len(self.all_datafiles) > 0, f"No {file_format} files found in data_dir matching file_format"
        if len(self.all_datafiles) == 1:
            # Single file -> no validation split
            self.validation_split = None

        self.all_datafiles = np.array(self.all_datafiles)
        
        logging.info(f"[DATA] Detected data format: {self.data_format}")
        logging.info(f"[DATA] Total files found: {len(self.all_datafiles)}")

        # Probe the first file to discover image shape (needed for set_shape after tf.numpy_function)
        if self.data_format == 'npz':
            sample_data = np.load(self.all_datafiles[0])
            sample_arr = sample_data[self.image_key]
            if len(sample_arr.shape) == 2:
                sample_arr = np.expand_dims(sample_arr, axis=-1)
            elif len(sample_arr.shape) == 4:
                sample_arr = sample_arr[0]
            self.img_shape = sample_arr.shape  # (H, W, C)
            logging.info(f"[DATA] Probed NPZ image shape: {self.img_shape}")

        # Train/valid split
        if self.validation_split is not None:
            nums_valid = int(len(self.all_datafiles) * self.validation_split)
            nums_valid = 1 if nums_valid == 0 else nums_valid
            nums_train = len(self.all_datafiles) - nums_valid
            nums_train = 1 if nums_train == 0 else nums_train
            idx = np.arange(len(self.all_datafiles))
            np.random.shuffle(idx)
            self.train_datafiles = self.all_datafiles[idx[0:nums_train]]
            self.valid_datafiles = self.all_datafiles[idx[nums_train:]]
        else:
            self.train_datafiles = self.all_datafiles
            self.valid_datafiles = []

        logging.info(f"[DATA] Found {len(self.train_datafiles)} training files")
        logging.info(f"[DATA] Found {len(self.valid_datafiles)} validation files.")

        # Base datasets (paths only)
        self.train_ds = tf.data.Dataset.list_files(self.train_datafiles)
        self.valid_ds = (
            tf.data.Dataset.list_files(self.valid_datafiles)
            if len(self.valid_datafiles) > 0 else
            None
        )

    def _tf_resize_and_center_crop(self, img):
        """
        Resize image to have the smaller dimension equal to img_resize,
        keeping aspect ratio, then center crop to [img_resize, img_resize].
        Pure TensorFlow operations for GPU acceleration.

        Args:
            img (tf.Tensor): Input image tensor of shape [H, W, C].
        Returns:
            tf.Tensor: Resized and center-cropped image tensor of shape [img_resize, img_resize, C].
        """
        target_size = self.img_resize
        
        # Get current shape
        shape = tf.shape(img)
        h, w = shape[0], shape[1]
        h_f = tf.cast(h, tf.float32)
        w_f = tf.cast(w, tf.float32)
        
        # Calculate new dimensions maintaining aspect ratio
        min_dim = tf.minimum(h_f, w_f)
        scale = tf.cast(target_size, tf.float32) / min_dim
        new_h = tf.cast(tf.round(h_f * scale), tf.int32)
        new_w = tf.cast(tf.round(w_f * scale), tf.int32)
        
        # Ensure dimensions are at least target_size
        new_h = tf.maximum(target_size, new_h)
        new_w = tf.maximum(target_size, new_w)
        
        # Resize
        img_resized = tf.image.resize(img, [new_h, new_w], method='bilinear', antialias=True)
        
        # Center crop to target_size
        img_cropped = tf.image.resize_with_crop_or_pad(img_resized, target_size, target_size)
        
        return img_cropped

    def _tf_apply_crop(self, img, is_training=True):
        """
        Apply the configured cropping strategy using TensorFlow operations.

        Args:
            img (tf.Tensor): Input image tensor of shape [H, W, C].
            is_training (bool): If True, randomization-enabled strategies may run differently.

        Returns:
            tf.Tensor: Cropped image. If crop_size is None, returns the original image.
        """
        if self.crop_size is None:
            return img

        crop_size = self.crop_size
        h, w, c = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2]
        # Pad if crop_size exceeds current dimensions
        pad_needed = tf.reduce_any([crop_size > h, crop_size > w])
        
        def pad_image():
            pad_h = tf.maximum(0, crop_size - h)
            pad_w = tf.maximum(0, crop_size - w)
            paddings = [
                [pad_h // 2, pad_h - pad_h // 2],
                [pad_w // 2, pad_w - pad_w // 2],
                [0, 0]
            ]
            return tf.pad(img, paddings, mode='REFLECT')
        
        img = tf.cond(pad_needed, pad_image, lambda: img)
        
        # Apply crop based on type
        if self.crop_type == 'center':
            return tf.image.resize_with_crop_or_pad(img, crop_size, crop_size)
        elif self.crop_type == 'random' and is_training:
            return tf.image.random_crop(img, size=[crop_size, crop_size, c])
        elif self.crop_type == 'corner':
            return self._tf_corner_crop(img, crop_size)
        else:
            # Default to center crop
            return tf.image.resize_with_crop_or_pad(img, crop_size, crop_size)
    
    def _tf_corner_crop(self, img, crop_size):
        """Crop [crop_size, crop_size] from a specific corner using TensorFlow."""
        shape = tf.shape(img)
        h, w = shape[0], shape[1]
        
        if self.crop_position == 'top_left':
            offset_h, offset_w = 0, 0
        elif self.crop_position == 'top_right':
            offset_h, offset_w = 0, w - crop_size
        elif self.crop_position == 'bottom_left':
            offset_h, offset_w = h - crop_size, 0
        elif self.crop_position == 'bottom_right':
            offset_h, offset_w = h - crop_size, w - crop_size
        else:
            # Fallback to center
            offset_h = (h - crop_size) // 2
            offset_w = (w - crop_size) // 2
        
        return tf.image.crop_to_bounding_box(img, offset_h, offset_w, crop_size, crop_size)

    def _tf_apply_augmentation(self, img):
        """
        Apply data augmentation using TensorFlow operations for GPU acceleration.

        Supported augment_type:
            - 'fliplr': Random horizontal flip
            - 'flipud': Random vertical flip
            - 'rotate': Random 90-degree rotation
            - 'flip-rotate': Combined random flip and rotation

        Args:
            img (tf.Tensor): Input image tensor [H, W, C]

        Returns:
            tf.Tensor: Augmented image tensor
        """
        if not self.augment:
            return img

        if self.augment_type == 'fliplr':
            img = tf.image.random_flip_left_right(img)
        elif self.augment_type == 'flipud':
            img = tf.image.random_flip_up_down(img)
        elif self.augment_type == 'rotate':
            # Random 90-degree rotation
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k=k)
        elif self.augment_type == 'flip-rotate':
            # Combined flip and rotate
            img = tf.image.random_flip_left_right(img)
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k=k)
        else:
            raise ValueError(f"Unknown augment_type: {self.augment_type}")

        return img

    def _tf_load_jpeg(self, path):
        """
        Load and decode a JPEG file using TensorFlow's optimized decoder.
        
        Args:
            path (tf.Tensor): Path to JPEG file (string tensor).
            
        Returns:
            tf.Tensor: Decoded image tensor [H, W, C] in float32 format [0, 1].
        """
        # Use TensorFlow's optimized JPEG decoder
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
        
        return img
    
    def _tf_load_png(self, path):
        """
        Load and decode a PNG file using TensorFlow's optimized decoder.
        Preserves original channel count: binary (1 channel), grayscale (1 channel), or RGB (3 channels).
        
        Args:
            path (tf.Tensor): Path to PNG file (string tensor).
            
        Returns:
            tf.Tensor: Decoded image tensor [H, W, C] in float32 format [0, 1].
                      C can be 1 (binary/grayscale) or 3 (RGB).
        """
        # Use TensorFlow's optimized PNG decoder with auto-detect channels
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_png(img_raw, channels=0)  # 0 = auto-detect channels
        
        # Convert to float32
        img = tf.cast(img, tf.float32)
        
        # Normalize to [0, 1] - handles both 8-bit (0-255) and binary (0-1) PNGs
        max_val = tf.reduce_max(img)
        img = tf.cond(
            max_val > 1.0,
            lambda: img / 255.0,  # 8-bit PNG (values 0-255)
            lambda: img            # Already binary or normalized (values 0-1)
        )
        
        return img

    def _load_data(self, path, is_training=True):
        """
        Load a single data file (.npz, .jpg/.jpeg, or .png) and apply preprocessing using TensorFlow operations.
          - Read image (and optional label for .npz).
          - All preprocessing uses TensorFlow ops for GPU acceleration.
          - Resize to [img_resize, img_resize] if img_resize provided.
          - Optional crop and augmentation.
          - Normalize to [-1, 1].

        Args:
            path (tf.Tensor): Path to a data file (string tensor).
            is_training (bool): Whether to apply training-time behaviors.

        Returns:
            tf.Tensor or (tf.Tensor, tf.Tensor): Image tensor (train_size, train_size, C),
            and label tensor (scalar) if label_key is set (NPZ only).
        """
        if self.data_format == 'jpg':
            # Load JPEG using pure TensorFlow operations
            img = self._tf_load_jpeg(path)
        elif self.data_format == 'png':
            # Load PNG using pure TensorFlow operations
            img = self._tf_load_png(path)
        
        if self.data_format in ['jpg', 'png']:
            # Apply preprocessing with TensorFlow ops for image files
            if self.img_resize is not None:
                img = self._tf_resize_and_center_crop(img)
            
            if self.crop_size is not None:
                img = self._tf_apply_crop(img, is_training=is_training)
            
            if is_training and self.augment:
                img = self._tf_apply_augmentation(img)
            
            # Normalize from [0, 1] to [-1, 1]
            img = img * 2.0 - 1.0
            
            # Set shape for known dimensions
            channel_dim = 3 if self.data_format == 'jpg' else None
            if self.crop_size is not None:
                img = tf.ensure_shape(img, [self.crop_size, self.crop_size, channel_dim])
            elif self.img_resize is not None:
                img = tf.ensure_shape(img, [self.img_resize, self.img_resize, channel_dim])
            
            return img
        else:
            # Load NPZ file (requires numpy_function for file I/O)
            def _load_npz(path_bytes):
                path_str = path_bytes.decode('utf-8')
                data = np.load(path_str)
                
                assert self.image_key in list(data.keys()), (
                    f"image_key '{self.image_key}' not found in {path_str}"
                )
                arr = data[self.image_key]
                
                # Normalize dimensionality to [H, W, C]
                if len(arr.shape) == 2:
                    arr = np.expand_dims(arr, axis=-1)
                elif len(arr.shape) == 4:
                    # If batch dimension exists, take first image
                    arr = arr[0]
                
                assert len(arr.shape) == 3, (
                    f"Array shape must be [H, W, C], got {arr.shape}"
                )
                
                arr = arr.astype(np.float32)
                
                # Load label if available
                label = np.int32(-1)
                if self.label_key is not None:
                    if self.label_key not in data:
                        raise KeyError(
                            f"label_key '{self.label_key}' not found in {path_str}"
                        )
                    label = np.array(data[self.label_key]).astype(np.int32)
                
                return arr, label
            
            # Load using numpy_function
            has_labels = (self.label_key is not None)
            img, label = tf.numpy_function(_load_npz, [path], [tf.float32, tf.int32])
            # numpy_function drops static shape info; restore using shape probed at __init__ time.
            img.set_shape(self.img_shape)
            
            # Apply preprocessing with TensorFlow ops
            if self.img_resize is not None:
                img = self._tf_resize_and_center_crop(img)
            
            if self.crop_size is not None:
                img = self._tf_apply_crop(img, is_training=is_training)
            
            if is_training and self.augment:
                img = self._tf_apply_augmentation(img)
            
            # Normalize from [0, 1] to [-1, 1]
            img = img * 2.0 - 1.0
            
            # Set shape
            if self.crop_size is not None:
                img = tf.ensure_shape(img, [self.crop_size, self.crop_size, None])
            elif self.img_resize is not None:
                img = tf.ensure_shape(img, [self.img_resize, self.img_resize, None])
            
            if has_labels:
                label = tf.ensure_shape(label, [])
                return img, label
            else:
                return img

    def _get_dataset(self):
        """
        Build training and validation datasets with appropriate preprocessing pipelines.

        Training dataset:
            - Maps file paths to preprocessed images (and labels).
            - If crop_type == 'random', caching is disabled to preserve per-epoch randomness.
            - Shuffles and repeats indefinitely.
            - Prefetches for performance.

        Validation dataset:
            - Deterministic preprocessing (is_training=False).
            - Cached and prefetched.
        """
        # When random cropping with cache enabled, cache the base decoded images (no crop),
        # then apply random crop/augmentation after repeat to avoid re-decoding large images.
        if (self.crop_size is not None) and self.crop_type == 'random' and self.cache:
            has_labels = (self.data_format == 'npz') and (self.label_key is not None)

            def _load_base(path):
                label = tf.constant(-1, dtype=tf.int32)
                if self.data_format == 'jpg':
                    img = self._tf_load_jpeg(path)
                elif self.data_format == 'png':
                    img = self._tf_load_png(path)
                else:
                    def _load_npz(path_bytes):
                        path_str = path_bytes.decode('utf-8')
                        data = np.load(path_str)
                        arr = data[self.image_key]
                        if len(arr.shape) == 2:
                            arr = np.expand_dims(arr, axis=-1)
                        elif len(arr.shape) == 4:
                            arr = arr[0]
                        arr = arr.astype(np.float32)
                        if has_labels:
                            if self.label_key not in data:
                                raise KeyError(f"label_key '{self.label_key}' not found in {path_str}")
                            label = np.array(data[self.label_key]).astype(np.int32)
                            return arr, label
                        return arr
                    if has_labels:
                        img, label = tf.numpy_function(_load_npz, [path], [tf.float32, tf.int32])
                        img.set_shape(self.img_shape)
                        label = tf.ensure_shape(label, [])
                        return img, label
                    img = tf.numpy_function(_load_npz, [path], tf.float32)
                    img.set_shape(self.img_shape)

                if self.img_resize is not None:
                    img = self._tf_resize_and_center_crop(img)

                if has_labels:
                    return img, label
                return img

            def _random_crop_aug_norm(img, label=None):
                # tf.data passes tuple elements as separate args when labels are present.
                if not has_labels:
                    label = None

                img = self._tf_apply_crop(img, is_training=True)
                if self.augment:
                    img = self._tf_apply_augmentation(img)
                img = img * 2.0 - 1.0

                channel_dim = 3 if self.data_format == 'jpg' else None
                img = tf.ensure_shape(img, [self.crop_size, self.crop_size, channel_dim])

                if has_labels:
                    return img, label
                return img

            train_ds = (
                self.train_ds
                .map(_load_base, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
                .shuffle(buffer_size=10000)
                .repeat()
                .map(_random_crop_aug_norm, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )
        elif (self.crop_size is not None) and self.crop_type == 'random':
            # Default random-crop path without caching cropped images
            train_ds = (
                self.train_ds
                .map(lambda x: self._load_data(x, is_training=True), num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(buffer_size=10000)
                .repeat()
                .prefetch(tf.data.AUTOTUNE)
            )
        elif self.cache:
            train_ds = (
                self.train_ds
                .map(lambda x: self._load_data(x, is_training=True), num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
                .shuffle(buffer_size=10000)
                .repeat()
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            train_ds = (
                self.train_ds
                .map(lambda x: self._load_data(x, is_training=True), num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(buffer_size=10000)
                .repeat()
                .prefetch(tf.data.AUTOTUNE)
            )
        options = tf.data.Options()
        options.experimental_deterministic = False
        train_ds = train_ds.with_options(options)

        if self.valid_ds is not None:
            valid_ds = (
                self.valid_ds
                .map(lambda x: self._load_data(x, is_training=False), num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            valid_ds = None

        return (train_ds, valid_ds)

    def get_info(self):
        """
        Return dataset configuration summary for quick inspection.

        Returns:
            dict: Keys include file counts, directory, keys, crop configuration, augmentation, and target size.
        """
        return {
            'data_format': self.data_format,
            'validation_split': self.validation_split,
            'num_train_files': len(self.train_datafiles),
            'num_valid_files': len(self.valid_datafiles),
            'total_files': len(self.all_datafiles),
            'data_dir': self.data_dir,
            'image_key': self.image_key,
            'label_key': self.label_key,
            'crop_type': self.crop_type,
            'crop_size': self.crop_size,
            'crop_position': self.crop_position,
            'augment': self.augment,
            'img_resize': self.img_resize,
            'cache': self.cache,
        }


def unit_test():
    """Unit test for DataLoader with enhanced cropping."""
    logging.info("[TEST] Running DataLoader unit test with enhanced cropping...")

    # create demo data directory for unit test, after that delete it
    import shutil
    demo_data_dir = "demo_data"
    if os.path.exists(demo_data_dir):
        shutil.rmtree(demo_data_dir)
    os.makedirs(demo_data_dir, exist_ok=True)
    # Create some dummy .npz files
    for i in range(10):
        # Random image size between 100 and 200
        h = np.random.randint(100, 200)
        w = np.random.randint(100, 200)
        img = np.random.rand(h, w, 3).astype(np.float32)
        label = np.random.randint(0, 10)
        npz_path = os.path.join(demo_data_dir, f"sample_{i}.npz")
        np.savez(npz_path, image=img, label=label)

    # Create a demo DataLoader instance
    data_loader = DataLoader(
        data_dir="demo_data",
        img_resize=128,
        crop_size=112,
        crop_type='random',
        augment=True,
        augment_type='flip-rotate',
        validation_split=0.2,
        image_key='image',
        label_key='label'
    )

    train_ds, valid_ds = data_loader._get_dataset()

    # Fetch a batch from training dataset
    for batch in train_ds.take(1):
        if data_loader.label_key is not None:
            images, labels = batch
            logging.info(f"[TEST] Train batch images shape: {images.shape}, labels shape: {labels.shape}")
        else:
            images = batch
            logging.info(f"[TEST] Train batch images shape: {images.shape}")

    # Fetch a batch from validation dataset
    if valid_ds is not None:
        for batch in valid_ds.take(1):
            if data_loader.label_key is not None:
                images, labels = batch
                logging.info(f"[TEST] Valid batch images shape: {images.shape}, labels shape: {labels.shape}")
            else:
                images = batch
                logging.info(f"[TEST] Valid batch images shape: {images.shape}")

    logging.info("[TEST] DataLoader unit test completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test DataLoader with enhanced cropping')
    parser.add_argument('--test', action='store_true', help='Run unit tests')

    args = parser.parse_args()

    if args.test:
        unit_test()

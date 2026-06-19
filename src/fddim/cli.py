"""
run.py
------
Main script for training and image generation using diffusion models.

Features:
- Configurable training and image generation via YAML config and command-line flags.
- Efficient dataset loading and prefetching.
- Model checkpointing, logging, and evaluation with FID.
- Supports XLA JIT compilation for performance.
"""

import os
import ast
import time
import datetime
import argparse
import shutil
import logging

from dataclasses import dataclass, fields
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import yaml
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from .diffusion_model import DiffusionModel
from .unet2d import build_model
from .diffusion_utils import DiffusionUtility
from .data_loader import DataLoader
from .callbacks import (
    WarmUpCosine,
    TQDMProgressBar,
    InlineImageGenerationCallback,
    BestModelCheckpoint,
    RobustCSVLogger,
)

# =====================
# Configuration Classes
# =====================

def dataclass_to_dict(obj):
    """Convert dataclass to dict for YAML serialization."""
    result = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, tuple):
            result[field.name] = list(value)  # Convert tuple to list for YAML
        else:
            result[field.name] = value
    return result


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    path: str
    label_key: Optional[str] = None
    img_resize: Optional[int] = None
    crop_size: Optional[int] = None
    crop_type: str = 'center'
    crop_position: str = 'center'
    augment: bool = False
    augment_type: Optional[str] = None
    cache: bool = False
    validation_split: Optional[float] = None

    def __init__(
        self,
        name: str,
        path: str,
        label_key: Optional[str] = None,
        img_resize: Optional[int] = None,
        crop_size: Optional[int] = None,
        crop_type: str = 'center',
        crop_position: str = 'center',
        augment: bool = False,
        augment_type: Optional[str] = None,
        cache: bool = False,
        validation_split: Optional[float] = None,
    ):
        self.name = name
        self.path = path
        self.label_key = label_key
        self.img_resize = img_resize
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.crop_position = crop_position
        self.augment = augment
        self.augment_type = augment_type
        self.cache = cache
        self.validation_split = validation_split


@dataclass
class DiffusionSchedulerConfig:
    """Diffusion scheduler configuration."""
    scheduler: str = 'cosine'  # Options: linear, cosine, my_cosine, my_cos6
    timesteps: int = 1000
    pred_type: str = 'velocity'  # Options: 'velocity', 'image', 'noise'

    def __init__(
        self,
        scheduler: str = 'cosine',
        timesteps: int = 1000,
        pred_type: str = 'velocity',
    ):
        self.scheduler = scheduler
        self.timesteps = timesteps
        self.pred_type = pred_type
    
    def to_yaml(self):
        """Convert to YAML string."""
        return yaml.dump(dataclass_to_dict(self), default_flow_style=False, sort_keys=False)
    
    def to_dict(self):
        """Convert to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class NetworkConfig:
    """Network architecture configuration."""
    image_size: Optional[Union[int, Tuple[int, int]]] # int or (int, int)
    image_channels: int
    base_channels: int = 64
    channel_multiplier: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    block_size: int = 1
    norm_groups: int = 32
    has_attention: Tuple[bool, ...] = (False, False, True, True)
    mid_attention: bool = True
    num_heads: int = 1
    embedding_type: str = 'positional'  # Options: 'positional', 'fourier'
    embedding_dim: Optional[int] = None
    time_emb_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_size: int = 3
    use_cross_attention: bool = False
    num_classes: Optional[int] = None
    class_emb_dim: Optional[int] = None
    skip_strategy: str = "per_block"  # Options: 'per_block' (default), 'stage' (lighter)

    def __init__(
        self,
        image_size: Optional[Union[int, Tuple[int, int]]],
        image_channels: int,
        base_channels: int = 64,
        channel_multiplier: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        block_size: int = 1,
        norm_groups: int = 32,
        has_attention: Tuple[bool, ...] = (False, False, True, True),
        mid_attention: bool = True,
        num_heads: int = 1,
        embedding_type: str = 'positional',
        embedding_dim: Optional[int] = None,
        time_emb_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        kernel_size: int = 3,
        use_cross_attention: bool = False,
        num_classes: Optional[int] = None,
        class_emb_dim: Optional[int] = None,
        skip_strategy: str = "per_block",
    ):
        self.image_size = image_size
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_res_blocks = num_res_blocks
        self.block_size = block_size
        self.norm_groups = norm_groups
        self.has_attention = has_attention
        self.mid_attention = mid_attention
        self.num_heads = num_heads
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.time_emb_dim = time_emb_dim
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.use_cross_attention = use_cross_attention
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.skip_strategy = skip_strategy
    
    def to_yaml(self):
        """Convert to YAML string."""
        return yaml.dump(dataclass_to_dict(self), default_flow_style=False, sort_keys=False)
    
    def to_dict(self):
        """Convert to dictionary."""
        return dataclass_to_dict(self)


@dataclass
class TrainingConfig:
    """Training configuration, including inline generation"""
    output_dir: str
    load_pretrained: Optional[str] = None
    loss_fn: str ='MSE'  # Options: 'MSE', 'MAE', 'BCE'
    loss_weight_type: str = 'min_snr'  # Options: 'constant', 'min_snr'
    min_snr_gamma: float = 5.0  # Gamma parameter for min-SNR weighting
    epochs: int = 100
    save_period: int = 10
    batch_size: int = 16
    grad_accum_steps: int = 1 # effective batch size = batch_size * grad_accum_steps
    steps_per_epoch: Optional[int] = None
    total_global_steps: Optional[int] = None
    ema: float = 0.999
    lr_type: str = 'constant'  # Options: 'constant', 'warmup_cosine'
    learning_rate: float = 1.0e-4
    warmup_steps: Optional[int] = None 
    inline_gen_enable: bool = True
    inline_gen_nums: int = 20
    inline_gen_period: int = 10
    inline_gen_reverse_steps: int = 100

    def __init__(
        self,
        output_dir: str,
        load_pretrained: Optional[str] = None,
        loss_fn: str = 'MSE',
        loss_weight_type: str = 'min_snr',
        min_snr_gamma: float = 5.0,
        epochs: int = 100,
        save_period: int = 10,
        batch_size: int = 16,
        grad_accum_steps: int = 1,
        steps_per_epoch: Optional[int] = None,
        total_global_steps: Optional[int] = None,
        ema: float = 0.999,
        lr_type: str = 'constant',  # Options: 'constant', 'warmup_cosine'
        learning_rate: float = 1.0e-4,
        warmup_steps: Optional[int] = None,
        inline_gen_enable: bool = True,
        inline_gen_nums: int = 20,
        inline_gen_period: int = 10,
        inline_gen_reverse_steps: int = 100,
    ):
        self.output_dir = output_dir
        self.load_pretrained = load_pretrained
        self.loss_fn = loss_fn
        self.loss_weight_type = loss_weight_type
        self.min_snr_gamma = min_snr_gamma
        self.epochs = epochs
        self.save_period = save_period
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.steps_per_epoch = steps_per_epoch
        self.total_global_steps = total_global_steps
        self.ema = ema
        self.lr_type = lr_type
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.inline_gen_enable = inline_gen_enable
        self.inline_gen_nums = inline_gen_nums
        self.inline_gen_period = inline_gen_period
        self.inline_gen_reverse_steps = inline_gen_reverse_steps


@dataclass
class ImageGenConfig:
    """(Post) Image generation configuration."""
    model_path: str
    gen_task: str = 'random'
    num_gen_images: int = 20
    batch_size: Optional[int] = None
    reverse_steps: int = 100
    ddim_eta: float = 1.0
    random_seed: Optional[int] = None
    target_image_size: Union[int, Tuple[int, int]] = None # int or (int, int)
    canvas_shape: Optional[Tuple[int, int]] = None  # for canvas_gen task only, (height, width)
    canvas_patch_size: Optional[int] = None  # for canvas_gen task only
    canvas_stride: Optional[int] = None  # for canvas_gen task only
    # output options
    save_dir: Optional[str] = None
    save_intermediate: bool = False
    save_format: str = 'png'  # Options: 'png', 'npz'
    # Optional parameters for specific tasks
    class_label: Optional[Union[int, List[int]]] = None
    freeze_channel: Optional[Union[int, List[int]]] = None
    space_inpaint_bbox: Optional[Tuple[int, int, int, int]] = None
    bbox_to_inpaint: bool = True
    external_input: Optional[str] = None
    clip_denoise: bool = False
    self_guide_scale: float = 0.0
    sdedit_strength: float = 0.5  # for img2img task only, strength of diffusion (0-1)
    overlap_dir: Optional[str] = None  # for overlap_inpaint task only, options: 'north', 'east', 'south', 'west'
    overlap_size: Optional[int] = None  # for overlap_inpaint task only, size of the overlap region

    def __init__(
        self,
        model_path: str,
        gen_task: str = 'random',
        num_gen_images: int = 20,
        batch_size: Optional[int] = None,
        reverse_steps: int = 100,
        ddim_eta: float = 1.0,
        random_seed: Optional[int] = None,
        target_image_size: Union[int, Tuple[int, int]] = None,
        canvas_shape: Optional[Tuple[int, int]] = None,
        canvas_patch_size: Optional[int] = None,
        canvas_stride: Optional[int] = None,
        save_dir: Optional[str] = None,
        save_intermediate: bool = False,
        save_format: str = 'png',
        class_label: Optional[Union[int, List[int]]] = None,
        freeze_channel: Optional[Union[int, List[int]]] = None,
        space_inpaint_bbox: Optional[Tuple[int, int, int, int]] = None,
        bbox_to_inpaint: bool = True,
        external_input: Optional[str] = None,
        clip_denoise: bool = False,
        self_guide_scale: float = 0.0,
        sdedit_strength: float = 0.5,
        overlap_dir: Optional[str] = None,
        overlap_size: Optional[int] = None,
    ):
        self.model_path = model_path
        self.gen_task = gen_task
        self.num_gen_images = num_gen_images
        self.batch_size = batch_size
        self.reverse_steps = reverse_steps
        self.ddim_eta = ddim_eta
        self.random_seed = random_seed
        self.target_image_size = target_image_size
        self.canvas_shape = canvas_shape
        self.canvas_patch_size = canvas_patch_size
        self.canvas_stride = canvas_stride
        self.save_dir = save_dir
        self.save_intermediate = save_intermediate
        self.save_format = save_format
        self.class_label = class_label
        self.freeze_channel = freeze_channel
        self.space_inpaint_bbox = space_inpaint_bbox
        self.bbox_to_inpaint = bbox_to_inpaint
        self.external_input = external_input
        self.clip_denoise = clip_denoise
        self.self_guide_scale = self_guide_scale
        self.sdedit_strength = sdedit_strength
        self.overlap_dir = overlap_dir
        self.overlap_size = overlap_size


# =====================
# Core Classes
# =====================

class ConfigManager:
    """Handles configuration parsing and validation."""
    
    @staticmethod
    def parse_config(
        config_path: str,
    ) -> Tuple[DatasetConfig, DiffusionSchedulerConfig, TrainingConfig, NetworkConfig, ImageGenConfig]:
        """Parse YAML config file into structured configs."""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Parse dataset config
        dataset_dict = cfg['DATASET']
        preprocessing = dataset_dict.get('PREPROCESSING', {})
        dataset_config = DatasetConfig(
            name=dataset_dict['NAME'],
            path=dataset_dict['PATH'],
            label_key=dataset_dict.get('LABEL_KEY'),
            img_resize=preprocessing.get('IMG_RESIZE'),
            crop_size=preprocessing.get('CROP_SIZE'),
            crop_type=preprocessing.get('CROP_TYPE', 'center'),
            crop_position=preprocessing.get('CROP_POSITION', 'center'),
            augment=preprocessing.get('AUGMENT', False),
            augment_type=preprocessing.get('AUGMENT_TYPE'),
            cache=preprocessing.get('CACHE', False),
            validation_split=preprocessing.get('VALIDATION_SPLIT'),
        )
        
        # Parse training config
        training_dict = cfg['TRAINING']
        inline_gen = training_dict.get('INLINE_GEN', {})
        hyper_params = training_dict['HYPER_PARAMETERS']
        training_config = TrainingConfig(
            output_dir=training_dict.get('OUTPUT_DIR', './training_outputs'),
            load_pretrained=training_dict.get('LOAD_PRETRAINED', None),
            loss_fn=training_dict.get('LOSS_FN', 'MSE'),
            loss_weight_type=training_dict.get('LOSS_WEIGHT_TYPE', 'min_snr'),
            min_snr_gamma=training_dict.get('MIN_SNR_GAMMA', 5.0),
            epochs=hyper_params['EPOCHS'],
            save_period=hyper_params.get('SAVE_PERIOD'),
            batch_size=hyper_params['BATCH_SIZE'],
            grad_accum_steps=hyper_params.get('GRAD_ACCUM_STEPS', 1),
            steps_per_epoch=hyper_params.get('STEPS_PER_EPOCH'),
            total_global_steps=hyper_params.get('TOTAL_GLOBAL_STEPS'),
            ema=hyper_params.get('EMA', 0.999),
            lr_type=hyper_params.get('LR_TYPE', 'constant'),
            learning_rate=hyper_params.get('LEARNING_RATE', 1.0e-4),
            warmup_steps=hyper_params.get('WARMUP_STEPS', None),
            inline_gen_enable=inline_gen.get('ENABLE', True),
            inline_gen_nums=inline_gen.get('NUMS', 20),
            inline_gen_period=inline_gen.get('PERIOD', 10),
            inline_gen_reverse_steps=inline_gen.get('REVERSE_STEPS', 100),
        )
        if training_config.grad_accum_steps < 1:
            raise ValueError("GRAD_ACCUM_STEPS must be >= 1")
        if training_config.total_global_steps is not None:
            if training_config.total_global_steps < 1:
                raise ValueError("TOTAL_GLOBAL_STEPS must be >= 1")
            if training_config.steps_per_epoch is not None:
                raise ValueError(
                    "Specify only one of TOTAL_GLOBAL_STEPS or STEPS_PER_EPOCH"
                )
            total_micro_steps = (
                training_config.total_global_steps
                * training_config.grad_accum_steps
            )
            if total_micro_steps % training_config.epochs != 0:
                raise ValueError(
                    "TOTAL_GLOBAL_STEPS * GRAD_ACCUM_STEPS must be divisible by EPOCHS"
                )
            training_config.steps_per_epoch = total_micro_steps // training_config.epochs
        else:
            if training_config.steps_per_epoch is None:
                raise ValueError(
                    "Either TOTAL_GLOBAL_STEPS or STEPS_PER_EPOCH must be provided"
                )
            if training_config.steps_per_epoch < 1:
                raise ValueError("STEPS_PER_EPOCH must be >= 1")
            training_config.total_global_steps = (
                training_config.epochs * training_config.steps_per_epoch
            ) // training_config.grad_accum_steps
        # Parse diffusion scheduler config
        diffusion_scheduler_dict = cfg['DIFFUSION_SCHEDULER']
        diffusion_scheduler_config = DiffusionSchedulerConfig(
            scheduler=diffusion_scheduler_dict.get('SCHEDULER', 'cosine'),
            timesteps=diffusion_scheduler_dict.get('TIMESTEPS', 1000),
            pred_type=diffusion_scheduler_dict.get('PRED_TYPE', 'velocity')
        )

        # Parse network config
        network_dict = cfg['NETWORK']
        network_config = NetworkConfig(
            image_size=network_dict.get('IMAGE_SIZE'),
            image_channels=network_dict['IMAGE_CHANNELS'],
            block_size=network_dict.get('BLOCK_SIZE', 1),
            num_res_blocks=network_dict.get('NUM_RES_BLOCKS', 2),
            norm_groups=network_dict.get('NORM_GROUPS', 32),
            base_channels=network_dict.get('BASE_CHANNELS', 64),
            channel_multiplier=network_dict.get('CHANNEL_MULTIPLIER', [1, 2, 4, 8]),
            has_attention=network_dict.get('HAS_ATTENTION', [False, False, True, True]),
            mid_attention=network_dict.get('MID_ATTENTION', True),
            num_heads=network_dict.get('NUM_HEADS', 1),
            embedding_type=network_dict.get('EMBEDDING_TYPE', 'positional'),
            embedding_dim=network_dict.get('EMBEDDING_DIM'),
            time_emb_dim=network_dict.get('TIME_EMB_DIM'),
            dropout_rate=network_dict.get('DROPOUT_RATE', 0.1),
            kernel_size=network_dict.get('KERNEL_SIZE', 3),
            use_cross_attention=network_dict.get('USE_CROSS_ATTENTION', False),
            num_classes=network_dict.get('NUM_CLASSES'),
            class_emb_dim=network_dict.get('CLASS_EMB_DIM'),
            skip_strategy=network_dict.get('SKIP_STRATEGY', 'per_block'), # Options: 'per_block', 'stage'
        )
        
        # Parse image generation config
        imgen_dict = cfg['IMAGE_GENERATION']
        imgen_outputs_dict = imgen_dict.get('OUTPUT_OPTIONS', {})
        imgen_cond_dict = imgen_dict.get('CONDITIONING', {})
        imgen_config = ImageGenConfig(
            model_path        = imgen_dict.get('MODEL_PATH', ''),
            gen_task          = imgen_dict.get('GEN_TASK', 'random'),
            num_gen_images    = imgen_dict.get('NUM_GEN_IMAGES', 20),
            batch_size        = imgen_dict.get('BATCH_SIZE'),
            reverse_steps     = imgen_dict.get('REVERSE_STEPS', 100),
            canvas_shape      = imgen_dict.get('CANVAS_SHAPE'),
            canvas_patch_size = imgen_dict.get('CANVAS_PATCH_SIZE'),
            canvas_stride     = imgen_dict.get('CANVAS_STRIDE'),
            ddim_eta          = imgen_dict.get('DDIM_ETA', 1.0),
            random_seed       = imgen_dict.get('RANDOM_SEED'),
            target_image_size = imgen_dict.get('TARGET_IMAGE_SIZE'),
            self_guide_scale  = imgen_dict.get('_SELF_GUIDE_SCALE', 0.0),
            sdedit_strength   = imgen_dict.get('_SDEDIT_STRENGTH', 0.5),
            clip_denoise      = imgen_dict.get('CLIP_DENOISE', False),
            save_dir            = imgen_outputs_dict.get('SAVE_DIR'),
            save_intermediate   = imgen_outputs_dict.get('SAVE_INTERMEDIATE', False),
            save_format         = imgen_outputs_dict.get('SAVE_FORMAT', 'png'),
            class_label         = imgen_cond_dict.get('CLASS_LABEL'),
            freeze_channel      = imgen_cond_dict.get('FREEZE_CHANNEL'),
            space_inpaint_bbox  = imgen_cond_dict.get('SPACE_INPAINT_BBOX'),
            bbox_to_inpaint     = imgen_cond_dict.get('BBOX_TO_INPAINT', True),
            external_input      = imgen_cond_dict.get('EXTERNAL_INPUT'),
            overlap_dir         = imgen_cond_dict.get('OVERLAP_DIR'),
            overlap_size        = imgen_cond_dict.get('OVERLAP_SIZE'),
        )
        configs = {}
        configs['DATASET'] = dataset_config
        configs['DIFFUSION_SCHEDULER'] = diffusion_scheduler_config
        configs['TRAINING'] = training_config
        configs['NETWORK'] = network_config
        configs['IMAGE_GENERATION'] = imgen_config
        return configs


class ModelBuilder:
    """Handles model construction and related utilities."""
    
    @staticmethod
    def build_models(network_config: NetworkConfig) -> Tuple[keras.Model, keras.Model]:
        """Build main and EMA models."""
        kwargs = dict(
            image_size=network_config.image_size,
            image_channels=network_config.image_channels,
            base_channels=network_config.base_channels,
            channel_multiplier=network_config.channel_multiplier,
            has_attention=network_config.has_attention,
            mid_attention=network_config.mid_attention,
            num_heads=network_config.num_heads,
            num_res_blocks=network_config.num_res_blocks,
            norm_groups=network_config.norm_groups,
            actf=keras.activations.swish,
            block_size=network_config.block_size,
            embedding_type=network_config.embedding_type,
            embedding_dim=network_config.embedding_dim,
            time_emb_dim=network_config.time_emb_dim,
            dropout_rate=network_config.dropout_rate,
            kernel_size=network_config.kernel_size,
            use_cross_attention=network_config.use_cross_attention,
            num_classes=network_config.num_classes,
            class_emb_dim=network_config.class_emb_dim,
            skip_strategy=network_config.skip_strategy,
        )
        
        network = build_model(**kwargs)
        ema_network = build_model(**kwargs)
        ema_network.set_weights(network.get_weights()) # initialize ema_network weights same as main network
        return network, ema_network
    
    @staticmethod
    def create_diffusion_utility(diffusion_scheduler_config: DiffusionSchedulerConfig, 
                                 reverse_steps=None, ddim_eta: float = 1.0, 
                                 clip_denoise: bool = False) -> DiffusionUtility:
        """Create diffusion utility with common parameters."""
        if reverse_steps is None:
            reverse_steps = diffusion_scheduler_config.timesteps
        return DiffusionUtility(
            timesteps=diffusion_scheduler_config.timesteps,
            scheduler=diffusion_scheduler_config.scheduler,
            pred_type=diffusion_scheduler_config.pred_type,
            reverse_steps=reverse_steps,
            ddim_eta=ddim_eta, 
            clip_denoise=clip_denoise
        )
    
    @staticmethod
    def create_lr_schedule(training_config: TrainingConfig):
        """Create learning rate schedule."""
        if training_config.lr_type == 'constant':
            return training_config.learning_rate
        elif training_config.lr_type == 'warmup_cosine':
            if training_config.total_global_steps is None:
                raise ValueError("TOTAL_GLOBAL_STEPS must be set for warmup_cosine")
            total_steps = training_config.total_global_steps
            if training_config.warmup_steps is None:
                training_config.warmup_steps = total_steps // 20
            return WarmUpCosine(
                base_lr=training_config.learning_rate,
                warmup_steps=training_config.warmup_steps,
                total_steps=total_steps
            )
        elif training_config.lr_type == 'cosine_decay':
            return keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=training_config.learning_rate,
                decay_steps=10000,
                alpha=0.0
            )
        else:
            raise NotImplementedError(f"Learning rate type {training_config.lr_type} not implemented")
    
    @staticmethod
    def create_loss_function(loss_fn_name: str):
        """Create loss function with reduction='none' for per-sample loss computation.
            reduction='none' is required for custom loss weighting (e.g., min-SNR).
        """
        if loss_fn_name == "MAE":
            return keras.losses.MeanAbsoluteError(reduction='none')
        elif loss_fn_name == 'MSE':
            return keras.losses.MeanSquaredError(reduction='none')
        elif loss_fn_name == 'BCE':
            return keras.losses.BinaryCrossentropy(reduction='none')
        else:
            raise NotImplementedError(f"Loss function {loss_fn_name} not implemented")


class DirectoryManager:
    """Handles directory creation and logging setup."""
    
    @staticmethod
    def init_logging(filename: str, checkpoint: Optional[str] = None):
        """Initialize logging to file and console."""
        mode = "w+"
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.FileHandler(filename, mode=mode), logging.StreamHandler()],
        )
    
    @staticmethod
    def setup_training_directories(
        dataset_config: DatasetConfig, 
        training_config: TrainingConfig, 
        network_config: NetworkConfig, 
        diffusion_scheduler_config: DiffusionSchedulerConfig, 
        ) -> str:
        """Set up training directories and logging."""
        os.makedirs(training_config.output_dir, exist_ok=True)
        # get input image size
        image_size = None
        if dataset_config.img_resize is not None:
            image_size = dataset_config.img_resize
        if dataset_config.crop_size is not None:
            image_size = dataset_config.crop_size
        if image_size is None:
            if network_config.image_size is not None:
                image_size = network_config.image_size
            else:
                raise ValueError("Cannot determine input image size from dataset or network config")
        image_size_tag = str(image_size) if isinstance(image_size, int) else f"{image_size[0]}x{image_size[1]}" 
        # Create model name tag
        model_nametag = f"unet{image_size_tag}s{network_config.block_size}-"
        model_nametag += f"w{network_config.base_channels}m{''.join(map(str, network_config.channel_multiplier))}"
        model_nametag += f"g{network_config.norm_groups}rb{network_config.num_res_blocks}"
        model_nametag += f"_ema{str(training_config.ema).replace('.', '')}"
        model_nametag += f"_cond{network_config.num_classes}" if network_config.num_classes else "_uncond"
        
        # Create dataset and model directories
        dataset_tag = f"{dataset_config.name}_{network_config.image_channels}x{image_size_tag}"
        if dataset_config.crop_size is not None:
            dataset_tag += f"_{dataset_config.crop_type}_crop"
        if dataset_config.augment:
            dataset_tag += f"_aug_{dataset_config.augment_type}"
        dataset_tag = os.path.join(
            os.path.abspath(training_config.output_dir), 
            dataset_tag
        )
        os.makedirs(dataset_tag, exist_ok=True)
        
        tr_output_dir = os.path.join(dataset_tag, model_nametag)
        os.makedirs(tr_output_dir, exist_ok=True)
        
        # Create timestamped directory
        dateID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dateID = "_".join([
            diffusion_scheduler_config.scheduler, 
            str(diffusion_scheduler_config.timesteps), 
            diffusion_scheduler_config.pred_type, 
            training_config.loss_fn, training_config.loss_weight_type,
            f"bs{training_config.batch_size}x{training_config.grad_accum_steps}",
            dateID])
        
        if training_config.load_pretrained is None:
            # new training
            logging_dir = os.path.join(tr_output_dir, dateID)
            os.makedirs(logging_dir, exist_ok=True)
            DirectoryManager.init_logging(os.path.join(logging_dir, "train.log"))
            logging.info("[INFO] Start a new training")
        else:
            pretrain_model_path = training_config.load_pretrained
            if not pretrain_model_path.endswith('.h5'):
                raise ValueError("Pretrained model path must be an h5 file")
            logging_dir = os.path.join(tr_output_dir, f"finetune_{dateID}")
            os.makedirs(logging_dir, exist_ok=True)
            DirectoryManager.init_logging(os.path.join(logging_dir, "train.log"))
            logging.info(f"[INFO] Restoring model from: {pretrain_model_path}")
            logging.info("[INFO] Continuous Transfer training ...")
        
        return logging_dir


class DatasetManager:
    """Handles dataset loading and preparation."""
    
    @staticmethod
    def prepare_datasets(
        dataset_config: DatasetConfig, 
        training_config: TrainingConfig,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Prepare training and validation datasets with efficient prefetching."""
        autotune = tf.data.AUTOTUNE
        train_ds, valid_ds = None, None
        dataloader = DataLoader(
            data_dir=dataset_config.path,
            img_resize=dataset_config.img_resize,
            crop_size=dataset_config.crop_size,
            crop_type=dataset_config.crop_type,
            crop_position=dataset_config.crop_position,
            augment=dataset_config.augment,
            augment_type=dataset_config.augment_type,
            label_key=dataset_config.label_key,
            file_format='auto',
            cache=dataset_config.cache,
            validation_split=dataset_config.validation_split,
        )
        train_ds, valid_ds = dataloader._get_dataset()
        
        # Batch and prefetch
        train_ds = train_ds.batch(training_config.batch_size, drop_remainder=True)
        train_ds = train_ds.prefetch(autotune)
        if valid_ds is not None:
            valid_ds = valid_ds.batch(training_config.batch_size)
            valid_ds = valid_ds.prefetch(autotune)
        
        return train_ds, valid_ds


class LoggingManager:
    """Handles structured logging."""
    
    @staticmethod
    def log_training_info(dataset_config: DatasetConfig, training_config: TrainingConfig, 
                         diffusion_scheduler_config: DiffusionSchedulerConfig):
        """Log comprehensive training information."""
        logging.info(f"[INFO] Training Start Time: {datetime.datetime.now()}")
        logging.info(f"[INFO] User defined dataset name: {dataset_config.name}")
        # Preprocessing info
        logging.info("[INFO] Preprocessing Configuration:")
        logging.info(f"  - Crop Size: {dataset_config.crop_size if dataset_config.crop_size else 'same as image size'}")
        logging.info(f"  - Crop Type: {dataset_config.crop_type}")
        logging.info(f"  - Crop Position: {dataset_config.crop_position}")
        logging.info(f"  - Data Augmentation: {dataset_config.augment}")
        # Training parameters
        logging.info(f"[INFO] Number of Noise Steps: {diffusion_scheduler_config.timesteps}")
        logging.info(f"[INFO] Noise Scheduler: {diffusion_scheduler_config.scheduler}")
        logging.info(f"[INFO] Learning Rate Type: {training_config.lr_type}")
        logging.info(f"[INFO] Learning Rate: {training_config.learning_rate}")
        logging.info(f"[INFO] Batch Size: {training_config.batch_size}")
        logging.info(f"[INFO] Gradient Accumulation Steps: {training_config.grad_accum_steps}")
        logging.info(
            f"[INFO] Effective Batch Size: "
            f"{training_config.batch_size * training_config.grad_accum_steps}"
        )
        if training_config.total_global_steps is not None:
            logging.info(
                f"[INFO] Total Global Steps (optimizer updates): "
                f"{training_config.total_global_steps}"
            )
        logging.info(f"[INFO] Predict Type: {diffusion_scheduler_config.pred_type}")
        logging.info(f"[INFO] Loss Function: {training_config.loss_fn}")
        logging.info(f"[INFO] Loss Weight Type: {training_config.loss_weight_type}")
        if training_config.loss_weight_type == 'min_snr':
            logging.info(f"[INFO] Min-SNR Gamma: {training_config.min_snr_gamma}")
        logging.info(f"[INFO] Total Epochs: {training_config.epochs}")
        logging.info(f"[INFO] Steps per Epoch (micro-batches): {training_config.steps_per_epoch}")
        logging.info(f"[INFO] EMA Decay Rate: {training_config.ema}")


# =====================
# Main Workflow Classes
# =====================

class DiffusionTrainer:
    """Handles the complete training workflow."""
    
    def __init__(self, config_file: str):
        # Setup GPU
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        configs = ConfigManager.parse_config(config_file)
        self.dataset_config = configs['DATASET']
        self.diffusion_scheduler_config = configs['DIFFUSION_SCHEDULER']
        self.training_config = configs['TRAINING']
        self.network_config = configs['NETWORK']
        self.config_file = config_file
        # Prepare datasets
        self.train_ds, self.valid_ds = DatasetManager.prepare_datasets(self.dataset_config, self.training_config)
        self.input_shape = None
        # get input shape from one batch
        for batch_data in self.train_ds.take(1):
            x = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            _, h, w, c = x.shape
            self.input_shape = (h, w, c)
        # Build models
        if h==w:
            self.network_config.image_size = h
        else:
            self.network_config.image_size = (h, w)
        if self.network_config.image_channels != self.input_shape[2]:
            raise ValueError(
                "NETWORK.IMAGE_CHANNELS does not match dataset channels "
                f"({self.network_config.image_channels} != {self.input_shape[2]})"
            )
        network, ema_network = ModelBuilder.build_models(self.network_config)
        self.network = network
        self.ema_network = ema_network
        # Show inputs, outputs and total parameters only, no display of model graph details
        model_encoded = keras.Model(
            inputs=ema_network.inputs, 
            outputs=ema_network(ema_network.inputs, training=False), 
            name="UNet2D")
        model_encoded.summary()

    def plot_model_graph(self):
        keras.utils.plot_model(
            self.network,
            to_file="unet_model_diagram.png",
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=120,
        )
        logging.info("Model diagram saved as unet_model_diagram.png")

    def train(self):
        """Execute the training workflow."""
        # Setup directories and logging
        logging_dir = DirectoryManager.setup_training_directories(
            self.dataset_config, 
            self.training_config, 
            self.network_config, 
            self.diffusion_scheduler_config,
        )
        # Copy config file
        shutil.copy(self.config_file, os.path.join(logging_dir, "model_config.yaml"))
        # save network_config to yaml
        with open(os.path.join(logging_dir, "network_config.yaml"), 'w') as f:
            f.write(self.network_config.to_yaml())
        # save diffusion_scheduler_config to yaml
        with open(os.path.join(logging_dir, "scheduler_config.yaml"), 'w') as f:
            f.write(self.diffusion_scheduler_config.to_yaml())
        
        # Create diffusion utilities and model
        diff_util_train = ModelBuilder.create_diffusion_utility(
            self.diffusion_scheduler_config, 
            clip_denoise=False,
        )
        
        dm = DiffusionModel(
            network=self.network,
            ema_network=self.ema_network,
            diff_util=diff_util_train,
            num_classes=self.network_config.num_classes,
            save_period=self.training_config.save_period,
            ema=self.training_config.ema,
            loss_weight_type=self.training_config.loss_weight_type,
            min_snr_gamma=self.training_config.min_snr_gamma,
            gradient_accumulation_steps=self.training_config.grad_accum_steps,
        )
        
        # Load existing model if continuing training
        if self.training_config.load_pretrained is not None:
            dm.ema_network.load_weights(self.training_config.load_pretrained)
            dm.network.set_weights(dm.ema_network.get_weights())
        
        # Log training information
        t0 = time.time()
        LoggingManager.log_training_info(self.dataset_config, self.training_config, self.diffusion_scheduler_config)
        
        # Setup training components
        lr_schedule = ModelBuilder.create_lr_schedule(self.training_config)
        loss_fn = ModelBuilder.create_loss_function(self.training_config.loss_fn)
        optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.0)
        
        # Setup callbacks
        callbacks = [
            RobustCSVLogger(os.path.join(logging_dir, "log.csv"), append=True, separator=","),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: dm.save_models(epoch, savedir=logging_dir)
            ),
            keras.callbacks.LambdaCallback(
                on_train_end=lambda logs: None  # Placeholder for generate_images_and_save
            ),
            TQDMProgressBar(),
        ]
        # Add callback to save the best EMA model weights by smallest loss
        callbacks.append(BestModelCheckpoint(
            filepath=os.path.join(logging_dir, "best_model"),
            monitor='val_loss' if self.valid_ds is not None else 'loss',
            mode='min',
            verbose=1
        ))

        if self.training_config.inline_gen_enable:
            callbacks.append(InlineImageGenerationCallback(
                period=self.training_config.inline_gen_period,
                num_images=self.training_config.inline_gen_nums,
                reverse_steps=self.training_config.inline_gen_reverse_steps,
                savedir=os.path.join(logging_dir, 'inline_gen'),
                labels=None,
            ))
        
        # Compile and train
        dm.compile(loss=loss_fn, optimizer=optimizer)
        interrupted = False
        try:
            dm.fit(
                self.train_ds,
                validation_data=self.valid_ds,
                epochs=self.training_config.epochs,
                steps_per_epoch=self.training_config.steps_per_epoch,
                callbacks=callbacks,
                verbose=0,  # Disable Keras default progress bar
            )
        except KeyboardInterrupt:
            interrupted = True
            gs = int(dm.optimizer.iterations.numpy())
            logging.warning("[TRAIN] Training interrupted by user (KeyboardInterrupt)")
            logging.info(f"[TRAIN] Global step at interrupt: {gs}")
        finally:
            delta_time = np.around((time.time() - t0) / 3600.0, 4)
            if interrupted:
                logging.info(
                    f"[TRAIN] Training stopped early at {datetime.datetime.now()}, "
                    f"elapsed time: {delta_time} hours"
                )
            else:
                logging.info(
                    f"[INFO] Training End: {datetime.datetime.now()}, "
                    f"elapsed time: {delta_time} hours"
                )
    

class ImageGenerator:
    """Handles the image generation workflow."""
    
    def __init__(self, config_file: str):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        configs = ConfigManager.parse_config(config_file)    
        self.config_file = config_file
        self.config_dir = os.path.dirname(config_file)
        self.dataset_config = configs['DATASET']
        self.diffusion_scheduler_config = configs['DIFFUSION_SCHEDULER']
        self.training_config = configs['TRAINING']
        self.network_config = configs['NETWORK']
        self.imgen_config = configs['IMAGE_GENERATION']
    
    def generate(self):
        """Execute the image generation workflow."""
        # Validate generation config
        if not self.imgen_config.gen_task:
            raise ValueError("IMAGE_GENERATION.GEN_TASK must be provided")
        if not self.imgen_config.model_path or not os.path.isfile(self.imgen_config.model_path):
            raise ValueError(
                f"IMAGE_GENERATION.MODEL_PATH does not exist: {self.imgen_config.model_path}"
            )
        
        # Setup generation directory
        model_dir = os.path.dirname(self.imgen_config.model_path)
        gen_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        gen_steps = str(self.imgen_config.reverse_steps)
        gen_save_dir1 = "_".join(["imgen", self.imgen_config.gen_task, gen_steps + "steps"])
        gen_save_dir2 = "_".join([gen_date, os.uname().nodename])
        
        if self.imgen_config.save_dir is None:
            self.imgen_config.save_dir = self.config_dir
        self.imgen_config.save_dir = os.path.join(self.imgen_config.save_dir, gen_save_dir1, gen_save_dir2)

        os.makedirs(self.imgen_config.save_dir, exist_ok=True)

        # Set random seed
        if self.imgen_config.random_seed is None:
            self.imgen_config.random_seed = np.random.randint(0, 1e6)
        tf.random.set_seed(self.imgen_config.random_seed)
        
        # Setup logging
        DirectoryManager.init_logging(os.path.join(self.imgen_config.save_dir, "imgen.log"))
        self._log_generation_info()
        
        # Setup generation parameters
        base_images, labels = self._prepare_generation_inputs()
        
        # Create diffusion utility for inference
        diff_util_infer = ModelBuilder.create_diffusion_utility(
            self.diffusion_scheduler_config,
            reverse_steps=self.imgen_config.reverse_steps,
            ddim_eta=self.imgen_config.ddim_eta,
            clip_denoise=self.imgen_config.clip_denoise,
        ) 
        
        self.network_config.image_size = self.imgen_config.target_image_size
        
        if base_images is not None:
            self.network_config.image_size = (base_images.shape[1], base_images.shape[2])
         
        # Build models
        _, ema_model = ModelBuilder.build_models(self.network_config)
        # Load model weights
        ema_model.load_weights(self.imgen_config.model_path)
        # Show inputs, outputs and total parameters only, no display of model graph details
        ema_model_encoded = keras.Model(
            inputs=ema_model.inputs, 
            outputs=ema_model(ema_model.inputs, training=False), 
            name="UNet2D")
        ema_model_encoded.summary()

        dm_infer = DiffusionModel(
            network=ema_model,
            ema_network=ema_model,
            diff_util=diff_util_infer,
            num_classes=self.network_config.num_classes,
        )
        
        # Generate images
        t0 = time.time()
        dm_infer.generate_images_and_save(
            gen_task=self.imgen_config.gen_task,
            num_images=self.imgen_config.num_gen_images,
            batch_size=self.imgen_config.batch_size,
            reverse_steps=self.imgen_config.reverse_steps,
            canvas_shape=self.imgen_config.canvas_shape,
            canvas_patch_size=self.imgen_config.canvas_patch_size,
            canvas_stride=self.imgen_config.canvas_stride,
            savedir=self.imgen_config.save_dir,
            clip_denoise=self.imgen_config.clip_denoise,
            base_images=base_images,
            labels=labels,
            inpaint_mask=None,
            freeze_channel=self.imgen_config.freeze_channel,
            space_inpaint_bbox=self.imgen_config.space_inpaint_bbox,
            bbox_to_inpaint=self.imgen_config.bbox_to_inpaint,
            save_intermediate=self.imgen_config.save_intermediate,
            save_format= self.imgen_config.save_format,
            self_guide_scale=self.imgen_config.self_guide_scale,
            sdedit_strength=self.imgen_config.sdedit_strength,
            overlap_dir=self.imgen_config.overlap_dir,
            overlap_size=self.imgen_config.overlap_size,
        )
        
        # Log completion
        delta_time = np.around((time.time() - t0), 1)
        logging.info(f"Generation images completed with {delta_time} seconds")
        logging.info(f"[IMGEN] {self.imgen_config.num_gen_images} images generated and saved to {self.imgen_config.save_dir}")
        logging.info(f"[IMGEN] image size {self.imgen_config.target_image_size}, channel {self.network_config.image_channels}")
        
    def _log_generation_info(self):
        """Log generation configuration."""
        logging.info(f"[IMGEN] Start to generate images using model: {self.imgen_config.model_path}")
        logging.info(f"[IMGEN] Generation Task: {self.imgen_config.gen_task}")
        if self.imgen_config.gen_task == 'img2img':
            logging.info(f"[IMGEN] SDEdit strength = {self.imgen_config.sdedit_strength}")
        logging.info(f"[IMGEN] External input: {self.imgen_config.external_input}")
        logging.info(f"[IMGEN] freeze channel: {self.imgen_config.freeze_channel}")
        logging.info(f"[IMGEN] class label: {self.imgen_config.class_label}")
        logging.info(f"[IMGEN] Model Predict Type: {self.diffusion_scheduler_config.pred_type}")
        logging.info(f"[IMGEN] DDIM eta = {self.imgen_config.ddim_eta}")
        logging.info(f"[IMGEN] self guide scale = {self.imgen_config.self_guide_scale}")
        logging.info(f"[IMGEN] Set Random Seed: {self.imgen_config.random_seed}")
        logging.info(f"[IMGEN] clip_denoise: {self.imgen_config.clip_denoise}")
        logging.info(f"[IMGEN] hostname: {os.uname().nodename}")
        logging.info(f"[IMGEN] TF version: {tf.__version__}")
    
    def _prepare_generation_inputs(self) -> Tuple[Optional[np.ndarray], Optional[tf.Tensor]]:
        """Prepare base images and labels for generation."""
        # Validate generation task
        if self.imgen_config.gen_task == "random" or self.imgen_config.gen_task == "canvas_gen":
            # null the setting of external_input if gen_task is random
            self.imgen_config.external_input = None
        elif self.imgen_config.gen_task == 'channel_inpaint':
            if self.imgen_config.external_input is None:
                raise ValueError("CONDITIONING.EXTERNAL_INPUT is required for channel_inpaint")
            if self.imgen_config.freeze_channel is None:
                raise ValueError("CONDITIONING.FREEZE_CHANNEL is required for channel_inpaint")
        elif self.imgen_config.gen_task == 'space_inpaint':
            if self.imgen_config.external_input is None:
                raise ValueError("CONDITIONING.EXTERNAL_INPUT is required for space_inpaint")
            if self.imgen_config.space_inpaint_bbox is None:
                raise ValueError("CONDITIONING.SPACE_INPAINT_BBOX is required for space_inpaint")
        elif self.imgen_config.gen_task == 'overlap_inpaint':
            if self.imgen_config.external_input is None:
                raise ValueError("CONDITIONING.EXTERNAL_INPUT is required for overlap_inpaint")
            if self.imgen_config.overlap_dir is None:
                raise ValueError("CONDITIONING.OVERLAP_DIR is required for overlap_inpaint")
            if self.imgen_config.overlap_size is None:
                raise ValueError("CONDITIONING.OVERLAP_SIZE is required for overlap_inpaint")
        elif self.imgen_config.gen_task == 'img2img':
            if self.imgen_config.external_input is None:
                raise ValueError("CONDITIONING.EXTERNAL_INPUT is required for img2img")
        else:
            raise NotImplementedError(f"Generation task {self.imgen_config.gen_task} not implemented")
        
        # Prepare base images
        base_images = None
        if self.imgen_config.external_input:
            _, ext = os.path.splitext(self.imgen_config.external_input)
            if ext == ".npz":
                npz_data = np.load(self.imgen_config.external_input)
                img_key = 'images' if 'images' in list(npz_data) else 'image'
                base_images = npz_data[img_key].astype(np.float32)
                if img_key == 'image':
                    base_images = np.stack([base_images]*self.imgen_config.num_gen_images, axis=0)
                self.imgen_config.num_gen_images = base_images.shape[0]
            elif ext == '.png' or ext == '.jpg' or ext == '.jpeg':
                base_images = Image.open(self.imgen_config.external_input)
                base_images = np.array(base_images).astype(np.float32) / 255.0
                if len(base_images.shape) == 2:
                    base_images = np.expand_dims(base_images, axis=-1)
                base_images = np.stack([base_images]*self.imgen_config.num_gen_images, axis=0)
                self.imgen_config.num_gen_images = base_images.shape[0]
            else:
                raise ValueError("Invalid base_images shape")
            # normalize images to [-1, 1]
            base_images = 2.0 * base_images - 1.0
            logging.info(f"[IMGEN] Use {self.imgen_config.external_input} as inpainting input")
            if len(base_images.shape) != 4:
                raise ValueError(
                    "External input must resolve to 4D tensor [N, H, W, C], "
                    f"got shape {base_images.shape}"
                )

        # Prepare labels
        labels = None
        if isinstance(self.imgen_config.class_label, int):
            labels = tf.fill([self.imgen_config.num_gen_images], int(self.imgen_config.class_label))
        elif isinstance(self.imgen_config.class_label, list):
            labels = self.imgen_config.class_label * self.imgen_config.num_gen_images
            labels = labels[:self.imgen_config.num_gen_images]
            labels = tf.constant(labels, tf.int32)
        elif isinstance(self.imgen_config.class_label, str):
            try:
                parsed = ast.literal_eval(self.imgen_config.class_label)
                if not isinstance(parsed, (list, tuple)):
                    raise ValueError
                if not all(isinstance(v, int) for v in parsed):
                    raise ValueError
                labels_list = list(parsed)
            except (ValueError, SyntaxError, TypeError) as exc:
                raise ValueError("class_label string must be a Python literal list/tuple of ints") from exc
            labels_list = labels_list * self.imgen_config.num_gen_images
            labels_list = labels_list[:self.imgen_config.num_gen_images]
            labels = tf.constant(labels_list, tf.int32)
            
        return base_images, labels


# =====================
# Main Entry Point
# =====================

def render_config_template() -> str:
        """Render a complete YAML template with comments for user guidance."""
        return """# Config template
# Keep top-level sections: DATASET, DIFFUSION_SCHEDULER, TRAINING, NETWORK, IMAGE_GENERATION

DATASET:
    NAME: # your dataset name
    PATH: # your dataset path contains image files (.jpg/.jpeg/.png) or .npz files with 'image' key
    LABEL_KEY: null # optional, specify the key for labels in .npz files if applicable, default is 'label', useful for class-conditional training
    PREPROCESSING:
        IMG_RESIZE: null         # int or null, if set, it applies a resize to the input images before cropping. The resizing is isotropic, i.e., it resizes the shorter edge to IMG_RESIZE while keeping the aspect ratio. If null, no resizing is applied.
        CROP_SIZE: null          # int or null, if set, it applies a crop to the input images (after resizing if IMG_RESIZE is set). If int, it crops a square region of CROP_SIZE x CROP_SIZE. If null, no cropping is applied and the original image size is used for training.
        CROP_TYPE: center        # center | random | corner
        CROP_POSITION: center    # center | top_left | top_right | bottom_left | bottom_right
        AUGMENT: false
        AUGMENT_TYPE: null       # fliplr | flipud | rotate | flip-rotate
        CACHE: false
        VALIDATION_SPLIT: null   # float in (0, 1), or null

DIFFUSION_SCHEDULER:
    SCHEDULER: linear          # linear | cosine | my_cosine | my_cos6
    TIMESTEPS: 1000
    PRED_TYPE: velocity        # velocity | image | noise

TRAINING:
    OUTPUT_DIR: ./training_outputs # output directory for training logs and model checkpoints
    LOAD_PRETRAINED: null      # path to .h5, or null
    LOSS_FN: MSE               # MSE | MAE | BCE
    LOSS_WEIGHT_TYPE: min_snr  # constant | min_snr
    MIN_SNR_GAMMA: 5.0
    HYPER_PARAMETERS:
        EPOCHS: 100         # total epochs to train
        SAVE_PERIOD: 10     # save model every SAVE_PERIOD epochs
        BATCH_SIZE: 16      # batch size for training
        GRAD_ACCUM_STEPS: 1 # gradient accumulation steps, effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
        STEPS_PER_EPOCH: 1000    # set this OR TOTAL_GLOBAL_STEPS
        TOTAL_GLOBAL_STEPS: null # set this OR STEPS_PER_EPOCH, if not set, total global steps = STEPS_PER_EPOCH * EPOCHS / GRAD_ACCUM_STEPS, it determines the total number of optimizer updates (global steps) for training, used for learning rate scheduling and training progress tracking
        EMA: 0.999    # exponential moving average decay rate for model weights
        LR_TYPE: constant   # constant | warmup_cosine | cosine_decay
        LEARNING_RATE: 1.0e-4
        WARMUP_STEPS: null # if lr_type is warmup_cosine, the number of warmup steps. If null, it defaults to total_global_steps // 20
    INLINE_GEN:
        ENABLE: true  # whether to enable inline image generation during training for monitoring progress
        NUMS: 20      # number of images to generate for each inline generation
        PERIOD: 10    # generate images every PERIOD epochs
        REVERSE_STEPS: 100   # number of reverse diffusion steps for inline generation, typically smaller than the steps used for final generation to save time

NETWORK:
    IMAGE_SIZE: null     # int or [H, W]; runtime may infer from data
    IMAGE_CHANNELS: 1    # typically 1 for grayscale, 3 for RGB, or arbitrary from npz; runtime may infer from data
    BLOCK_SIZE: 1
    NUM_RES_BLOCKS: 2
    NORM_GROUPS: 32
    BASE_CHANNELS: 64
    CHANNEL_MULTIPLIER: [1, 2, 4, 8]
    HAS_ATTENTION: [false, false, true, true]
    MID_ATTENTION: true
    NUM_HEADS: 1
    EMBEDDING_TYPE: positional # positional | fourier
    EMBEDDING_DIM: null
    TIME_EMB_DIM: null
    DROPOUT_RATE: 0.1
    KERNEL_SIZE: 3
    USE_CROSS_ATTENTION: false
    NUM_CLASSES: null
    CLASS_EMB_DIM: null
    SKIP_STRATEGY: per_block   # per_block | stage

IMAGE_GENERATION:
    MODEL_PATH: ""   # path to the trained model .h5 file for image generation
    GEN_TASK: random           # random | canvas_gen | channel_inpaint | space_inpaint | overlap_inpaint | img2img
    NUM_GEN_IMAGES: 20
    BATCH_SIZE: null
    REVERSE_STEPS: 100
    CANVAS_SHAPE: null         # [H, W] for canvas_gen
    CANVAS_PATCH_SIZE: null
    CANVAS_STRIDE: null
    DDIM_ETA: 1.0
    RANDOM_SEED: null
    TARGET_IMAGE_SIZE: null    # int or [H, W]
    _SELF_GUIDE_SCALE: 0.0
    _SDEDIT_STRENGTH: 0.5
    CLIP_DENOISE: false       # inference only; clip predicted x0 during reverse sampling
    OUTPUT_OPTIONS:
        SAVE_DIR: null
        SAVE_INTERMEDIATE: false
        SAVE_FORMAT: png         # png | npz
    # advanced conditioning options for image generation tasks like inpainting or img2img, these are optional and can be null if not applicable for the gen_task
    CONDITIONING:
        CLASS_LABEL: null        # int | [int, ...] | null
        FREEZE_CHANNEL: null     # int | [int, ...] | null
        SPACE_INPAINT_BBOX: null # [r0, r1, c0, c1] or null
        BBOX_TO_INPAINT: true
        EXTERNAL_INPUT: null     # path to .npz/.png/.jpg/.jpeg
        OVERLAP_DIR: null        # north | east | south | west
        OVERLAP_SIZE: null
"""


def emit_config_template(output_path: Optional[str] = None):
    """Write YAML template to file, or print to stdout when output_path is empty."""
    content = render_config_template()
    if output_path and output_path != '-':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Config template written to: {output_path}")
    else:
        print(content)

def main():
    """Main entry point for training or image generation."""
    internal_debug = os.getenv("ENABLE_PLOT_MODEL_GRAPH", "0") == "1"
    parser = argparse.ArgumentParser(description="Fast DDIM Training and Image Generation")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--get-template",
        nargs='?',
        const='-',
        default=None,
        metavar='OUTPUT',
        help="Generate YAML config template. Print to stdout by default, or provide output file path.",
    )
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--training", action='store_true', help="Run training mode")
    mode_group.add_argument("--imgen", action='store_true', help="Run image generation mode")
    if internal_debug:
        mode_group.add_argument("--plot_model_graph", action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--enable_xla", action='store_true', help='Enable XLA JIT compilation')
    
    args = parser.parse_args()
    
    # Enable XLA if requested
    if args.enable_xla:
        tf.config.optimizer.set_jit(True)

    # Emit template and exit without requiring config/mode.
    if args.get_template is not None:
        emit_config_template(args.get_template)
        return

    if not args.config:
        parser.error("--config is required unless --get-template is used")

    if not (args.training or args.imgen or getattr(args, "plot_model_graph", False)):
        parser.error("One mode flag is required: --training or --imgen")
    
    # Execute requested mode
    if args.training:
        trainer = DiffusionTrainer(args.config)
        trainer.train()
    elif args.imgen:
        generator = ImageGenerator(args.config)
        generator.generate()
    elif getattr(args, "plot_model_graph", False):
        trainer = DiffusionTrainer(args.config)
        trainer.plot_model_graph()

if __name__ == "__main__":
    main()

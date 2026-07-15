"""
unet2d.py
--------
Defines a configurable 2D UNet model for image-to-image tasks with optional attention and time embedding support.

Functions:
    build_model(...):
        Builds and returns a Keras 2D UNet model with skip connections, residual blocks, and optional attention.

Example usage:
    from unet import build_model
    model = build_model(image_size=512, image_channels=3)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore

try:
    from .layers import (
        kernel_init,
        TimeEmbedding,
        TimeMLP,
        ResidualBlock,
        DownSample,
        UpSample,
        SpaceToDepth,
        DepthToSpace,
    )
except ImportError:
    # Allow direct script execution: `python src/fddim/unet2d.py`
    from layers import (
        kernel_init,
        TimeEmbedding,
        TimeMLP,
        ResidualBlock,
        DownSample,
        UpSample,
        SpaceToDepth,
        DepthToSpace,
    )


# Network benchmark placeholder presets.
# These presets approximate published ADM model settings for common resolutions.
NETWORK_BKM_PLACEHOLDER = {
    "DDPM256": {
        "image_size": 256,
        "image_channels": 3,
        "base_channels": 128,
        "channel_multiplier": (1, 1, 2, 2, 4, 4),
        "has_attention": (False, False, False, False, True, True),
        "mid_attention": True,
        "num_heads": 8,
        "num_res_blocks": 2,
        "norm_groups": 32,
        "interpolation": "nearest",
        "block_size": 1,
        "embedding_type": "positional",
        "dropout_rate": 0.1,
        "kernel_size": 3,
        "use_cross_attention": False,
        "num_classes": None,
        "class_emb_dim": None,
        "skip_strategy": "per_block",
    },
    "ADM64": {
        "image_size": 64,
        "image_channels": 3,
        "base_channels": 192,
        "channel_multiplier": (1, 2, 3, 4),
        "has_attention": (False, True, True, True),
        "mid_attention": True,
        "num_heads": 4,
        "num_res_blocks": 3,
        "norm_groups": 32,
        "interpolation": "nearest",
        "block_size": 1,
        "embedding_type": "positional",
        "dropout_rate": 0.1,
        "kernel_size": 3,
        "use_cross_attention": False,
        "num_classes": None,
        "class_emb_dim": None,
        "skip_strategy": "per_block",
    },
    "ADM256": {
        "image_size": 256,
        "image_channels": 3,
        "base_channels": 256,
        "channel_multiplier": (1, 1, 2, 2, 4, 4),
        "has_attention": (False, False, False, True, True, True),
        "mid_attention": True,
        "num_heads": 4,
        "num_res_blocks": 2,
        "norm_groups": 32,
        "interpolation": "nearest",
        "block_size": 1,
        "embedding_type": "positional",
        "dropout_rate": 0.0,
        "kernel_size": 3,
        "use_cross_attention": False,
        "num_classes": None,
        "class_emb_dim": None,
        "skip_strategy": "per_block",
    },
}


def build_model_from_bkm(name: str) -> keras.Model:
    """Build a model from a named BKM placeholder preset."""
    if name not in NETWORK_BKM_PLACEHOLDER:
        valid = ", ".join(sorted(NETWORK_BKM_PLACEHOLDER.keys()))
        raise ValueError(f"Unknown BKM preset '{name}'. Available presets: {valid}")
    return build_model(**NETWORK_BKM_PLACEHOLDER[name])

def build_model(
    image_size=256,
    image_channels=3,
    coordinate_conditioning=False,
    base_channels=128,
    channel_multiplier=(1, 1, 2, 2, 4, 4),
    has_attention=(False, False, False, False, True, False),
    mid_attention=True,
    num_heads=1,
    num_res_blocks=2,
    norm_groups=32,
    interpolation="nearest",
    actf=keras.activations.swish,
    block_size=1,
    embedding_type="positional",  # 'positional' or 'fourier'
    embedding_dim=None,
    time_emb_dim=None,
    dropout_rate=0.1,
    kernel_size=3,
    use_cross_attention=False,
    num_classes=None,
    class_emb_dim=None,
    skip_strategy="per_block",  # 'per_block' (default) or 'stage' (lighter)
):
    """
    Build a configurable UNet model with skip connections, residual blocks, and optional attention.

    Parameters
    ----------
    image_size : int or tuple[int, int] or None
        Height and width of the input image. If int, assumes square image.
    image_channels : int
        Number of denoised image channels. Coordinate-conditioned models
        accept two additional input channels while retaining this output size.
    coordinate_conditioning : bool
        If True, reserve two input channels for normalized absolute ``(x, y)``
        coordinates as used by Patch Diffusion.
    base_channels : int
        Base number of channels for the first level of the UNet.
    channel_multiplier : list[int | float]
        Channel multipliers for each level of the UNet. Length determines number of levels.
    has_attention : list[bool]
        Whether to apply self-attention at each level. Length must match widths.
    num_heads : int
        Number of attention heads when attention is enabled.
    num_res_blocks : int
        Number of residual blocks at each level.
    norm_groups : int
        Number of groups for GroupNormalization.
    interpolation : str
        Upsampling interpolation method.
    actf : Callable
        Activation function used throughout the network.
    block_size : int
        Space-to-depth scaling factor.
    embedding_type : str {"positional", "fourier"}
        Type of time embedding to use.
    embedding_dim : int or None
        Dimension of the time embedding input. If None, defaults to
        ``widths[0]`` for 'positional' and ``2 * widths[0]`` for 'fourier'.
    time_emb_dim : int or None
        Dimension of the time embedding. If None, defaults to ``4 * widths[0]``.
    dropout_rate : float
        Dropout rate applied inside residual blocks.
    kernel_size : int or tuple[int, int]
        Convolution kernel size used for all convolutions.
    use_cross_attention : bool
        If True, attention blocks use cross-attention with the time embeddings.
    num_classes : int or None
        Number of classes for class conditioning. If provided, a label input is
        added and combined with the time embedding.
    class_emb_dim : int or None
        Dimension of the class embedding. Defaults to ``time_emb_dim`` when
        ``num_classes`` is specified.

    Additional Parameters
    ---------------------
    skip_strategy : str {"per_block", "stage"}
        Strategy for storing skip connections.
        per_block: push every residual block output + post-downsample (default, diffusers DDPM like).
        stage:     store only the final output of the residual stack at each resolution (Variant A memory saving).

    Returns
    -------
    keras.Model
        Model that maps [image, timestep] inputs to an image tensor of
        shape (batch, image_height, image_width, image_channels).
    """
    # Validate inputs
    if not isinstance(image_channels, int) or image_channels <= 0:
        raise ValueError("`image_channels` must be a positive integer.")
    if not isinstance(coordinate_conditioning, bool):
        raise ValueError("`coordinate_conditioning` must be a boolean.")
    if not isinstance(base_channels, int) or base_channels <= 0:
        raise ValueError("`base_channels` must be a positive integer.")
    if not isinstance(channel_multiplier, (list, tuple)) or len(channel_multiplier) == 0:
        raise ValueError("`channel_multiplier` must be a non-empty list/tuple of numbers.")
    if not all(isinstance(w, (int, float)) and w > 0 for w in channel_multiplier):
        raise ValueError("All elements in `channel_multiplier` must be positive numbers.")
    if not isinstance(has_attention, (list, tuple)) or len(has_attention) != len(channel_multiplier):
        raise ValueError("`has_attention` must be a list/tuple of booleans with the same length as `channel_multiplier`.")
    if not all(isinstance(h, bool) for h in has_attention):
        raise ValueError("All elements in `has_attention` must be booleans.")
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("`num_heads` must be a positive integer.")
    if not isinstance(num_res_blocks, int) or num_res_blocks <= 0:
        raise ValueError("`num_res_blocks` must be a positive integer.")
    if not isinstance(norm_groups, int) or norm_groups <= 0:
        raise ValueError("`norm_groups` must be a positive integer.")
    if not isinstance(interpolation, str):
        raise ValueError("`interpolation` must be a string.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("`block_size` must be a positive integer.")
    if time_emb_dim is not None and (not isinstance(time_emb_dim, int) or time_emb_dim <= 0):
        raise ValueError("`time_emb_dim` must be a positive integer.")
    if not isinstance(dropout_rate, (int, float)) or not 0 <= dropout_rate <= 1:
        raise ValueError("`dropout_rate` must be between 0 and 1.")
    if isinstance(kernel_size, int):
        if kernel_size <= 0:
            raise ValueError("`kernel_size` must be positive.")
        kernel_size = (kernel_size, kernel_size)
    elif (
        isinstance(kernel_size, (list, tuple))
        and len(kernel_size) == 2
        and all(isinstance(k, int) and k > 0 for k in kernel_size)
    ):
        kernel_size = tuple(kernel_size)
    else:
        raise ValueError("`kernel_size` must be an int or tuple of two ints.")
    if not isinstance(use_cross_attention, bool):
        raise ValueError("`use_cross_attention` must be a boolean.")
    
    # get input shape
    if isinstance(image_size, int):
        image_height = image_size
        image_width = image_size
    else:
        image_height, image_width = image_size
    widths = [int(round(base_channels * m)) for m in channel_multiplier]
    if not all(w > 0 for w in widths):
        raise ValueError("Computed stage widths must be positive; check base_channels and channel_multiplier.")
    input_channels = image_channels + 2 if coordinate_conditioning else image_channels
    input_shape = (image_height, image_width, input_channels)
    image_input = keras.Input(shape=input_shape, name="image_input") # tf.Tensor of shape (batch, H, W, C), float32
    time_input = keras.Input(shape=(), name="time_input") # tf.Tensor of shape (batch,), support both int32 (DDPM) and float32 (EDM)

    # Space-to-depth if block_size > 1
    if block_size > 1:
        assert image_height % block_size == 0
        assert image_width % block_size == 0
        x = SpaceToDepth(block_size)(image_input)
    else:
        x = image_input

    # Initial convolution
    x = keras.layers.Conv2D(
        filters=widths[0],
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(x)

    # Time embedding
    if embedding_dim is None:
        if embedding_type == "positional":
            embedding_dim_ = widths[0]
        elif embedding_type == "fourier":
            embedding_dim_ = 2 * widths[0]
        else:
            raise ValueError("embedding_type must be 'positional' or 'fourier'")
    else:
        embedding_dim_ = embedding_dim
    if time_emb_dim is None:
        time_emb_dim_ = 4 * widths[0]
    else:
        time_emb_dim_ = time_emb_dim
    temb = TimeEmbedding(dim=embedding_dim_, name="TimeEmb")(time_input)
    temb = TimeMLP(units=time_emb_dim_, actf=actf)(temb)

    inputs = [image_input, time_input]
    if num_classes is not None:
        if class_emb_dim is None:
            class_emb_dim_ = time_emb_dim_
        else:
            class_emb_dim_ = class_emb_dim
        class_input = keras.Input(shape=(), dtype=tf.int32, name="class_input")
        # set input_dim of class Embedding layer to "num_classes+1"
        # so the class integer input range: [0, num_classes+1)
        # integer 0 is reserved for null class (unconditioned)
        cemb = keras.layers.Embedding(num_classes+1, class_emb_dim_)(class_input)
        cemb = TimeMLP(units=time_emb_dim_, actf=actf)(cemb)
        temb = keras.layers.Add()([temb, cemb])
        inputs.append(class_input)
    else:
        class_input = None

    if skip_strategy not in {"per_block", "stage"}:
        raise ValueError("skip_strategy must be 'per_block' or 'stage'")

    skips = []

    # Downsampling path (two modes)
    if skip_strategy == "per_block":
        # Legacy behavior: push after each residual block and each downsample
        skips.append(x)
        for i in range(len(widths)):
            for _ in range(num_res_blocks):
                x = ResidualBlock(
                    out_channels=widths[i],
                    attention=has_attention[i],
                    num_heads=num_heads,
                    groups=norm_groups,
                    actf=actf,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size,
                    use_cross_attention=use_cross_attention,
                )([x, temb])
                skips.append(x)
            if i != len(widths) - 1:
                x = DownSample(widths[i], mode='conv')(x)
                skips.append(x)
    else:  # stage-level skips
        for i in range(len(widths)):
            for _ in range(num_res_blocks):
                x = ResidualBlock(
                    widths[i],
                    has_attention[i],
                    num_heads=num_heads,
                    groups=norm_groups,
                    actf=actf,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size,
                    use_cross_attention=use_cross_attention,
                )([x, temb])
            # After finishing residual stack at this resolution, store skip
            skips.append(x)
            if i != len(widths) - 1:
                x = DownSample(widths[i], mode='conv')(x)

    # Bottleneck
    x = ResidualBlock(
        widths[-1],
        mid_attention,
        num_heads=num_heads,
        groups=norm_groups,
        actf=actf,
        dropout_rate=dropout_rate,
        kernel_size=kernel_size,
        use_cross_attention=use_cross_attention,
        name="mid_resblock_1",
    )([x, temb])
    x = ResidualBlock(
        widths[-1],
        False,
        num_heads=num_heads,
        groups=norm_groups,
        actf=actf,
        dropout_rate=dropout_rate,
        kernel_size=kernel_size,
        use_cross_attention=use_cross_attention,
        name="mid_resblock_2",
    )([x, temb])

    # Upsampling path
    if skip_strategy == "per_block":
        for i in reversed(range(len(widths))):
            for _ in range(num_res_blocks + 1):
                x = keras.layers.Concatenate(axis=-1)([x, skips.pop()])
                x = ResidualBlock(
                    widths[i],
                    has_attention[i],
                    num_heads=num_heads,
                    groups=norm_groups,
                    actf=actf,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size,
                    use_cross_attention=use_cross_attention,
                )([x, temb])
            if i != 0:
                x = UpSample(widths[i], interpolation=interpolation)(x)
    else:  # stage-level decode
        for i in reversed(range(len(widths))):
            x = keras.layers.Concatenate(axis=-1)([x, skips.pop()])
            # Optionally project back to target width after concat
            x = keras.layers.Conv2D(
                widths[i], 1, padding="same", kernel_initializer=kernel_init(1.0), name=f"merge_proj_{i}"
            )(x)
            for _ in range(num_res_blocks):
                x = ResidualBlock(
                    widths[i],
                    has_attention[i],
                    num_heads=num_heads,
                    groups=norm_groups,
                    actf=actf,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size,
                    use_cross_attention=use_cross_attention,
                )([x, temb])
            if i != 0:
                x = UpSample(widths[i-1], interpolation=interpolation)(x)

    # Final normalization and convolution
    x = keras.layers.GroupNormalization(groups=norm_groups)(x)
    x = keras.layers.Activation(actf)(x)
    x = keras.layers.Conv2D(
        image_channels * (block_size ** 2),
        kernel_size,
        padding="same",
        kernel_initializer=kernel_init(0.0),
        name="final_conv2d",
    )(x)

    # Depth-to-space if block_size > 1
    if block_size > 1:
        x = DepthToSpace(block_size)(x)

    return keras.Model(inputs, x, name="unet")


def unit_test_build_model_defaults():
    """Build default model, print summary, and print total trainable weights."""
    model = build_model()
    model.summary()
    total_trainable = int(np.sum([np.prod(p.shape) for p in model.trainable_weights]))
    print(f"Total number of trainable weights: {total_trainable:,}")


def unit_test_build_bkm_network(name: str):
    """Build one BKM preset and print summary and trainable weight count."""
    print(f"\n===== BKM: {name} =====")
    model = build_model_from_bkm(name)
    model.summary()
    total_trainable = int(np.sum([np.prod(p.shape) for p in model.trainable_weights]))
    print(f"[{name}] Total number of trainable weights: {total_trainable:,}")


def unit_test_build_bkm_networks():
    """Run unit tests for all placeholder BKM presets."""
    for name in ["DDPM256", "ADM64", "ADM256"]:
        unit_test_build_bkm_network(name)


if __name__ == "__main__":
    unit_test_build_bkm_networks()

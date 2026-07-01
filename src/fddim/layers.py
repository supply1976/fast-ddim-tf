import tensorflow as tf
from tensorflow import keras # type: ignore


def get_activation(name):
    """Safely get activation function by name."""
    if isinstance(name, str):
        return getattr(keras.activations, name)
    return name


def kernel_init(scale):
    """
    Returns a Keras VarianceScaling initializer with the given scale.
    Args:
        scale (float): Scaling factor for the initializer. Must be positive.
    Returns:
        keras.initializers.VarianceScaling: The initializer instance.
    Raises:
        ValueError: If scale is not a number or is negative.
    """
    if not isinstance(scale, (int, float)):
        raise ValueError("scale must be a number.")
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")


class TimeEmbedding(keras.layers.Layer):
    """
    Sinusoidal time embedding layer.
    This align original DDPM positional embedding scheme.
    Args:
        dim (int): The embedding dimension. Must be a positive even integer.
    Input shape:
        1D tensor of shape (batch,)
    Output shape:
        2D tensor of shape (batch, dim)
    """
    def __init__(self, dim, max_period=10000.0, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(dim, int) or dim <= 0 or dim % 2 != 0:
            raise ValueError("`dim` must be a positive even integer.")
        self.dim = dim
        self.max_period = max_period

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        if len(inputs.shape) != 1:
            raise ValueError("Input tensor must be 1D (batch,). Got shape: {}".format(inputs.shape))

        half_dim = self.dim // 2
        freqs = tf.exp(-tf.math.log(self.max_period) * 
                      tf.range(start=0, limit=half_dim, dtype=tf.float32) / (half_dim-1))
        args = inputs[:, None] * freqs[None, :]
        embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return embedding

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "max_period": self.max_period})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GaussianFourierEmbedding(keras.layers.Layer):
    """
    Gaussian Fourier embedding layer for time or noise embeddings.
    Args:
        dim (int): The embedding dimension. Must be a positive even integer.
        scale (float): Scaling factor for the Gaussian matrix.
    Input shape:
        1D tensor of shape (batch,)
    Output shape:
        2D tensor of shape (batch, dim)
    """
    def __init__(self, dim, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(dim, int) or dim <= 0 or dim % 2 != 0:
            raise ValueError("`dim` must be a positive even integer.")
        self.dim = dim
        self.scale = scale

    def build(self, input_shape):
        # Create Gaussian random matrix
        self.W = self.add_weight(
            shape=(self.dim // 2,),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=self.scale),
            trainable=False,
            name="W",
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        if len(inputs.shape) != 1:
            raise ValueError("Input tensor must be 1D (batch,). Got shape: {}".format(inputs.shape))

        args = inputs[:, None] * self.W[None, :] * 2.0 * tf.constant(tf.math.pi)
        embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return embedding

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "scale": self.scale})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TimeMLP(keras.layers.Layer):
    """Two-layer MLP for processing time (or class) embeddings.

    Args:
        units (int): Hidden/output dimensionality.
        actf (callable|str): Activation for first Dense layer.
    Input shape:
        2D tensor (batch, dim)
    Output shape:
        2D tensor (batch, units)
    """
    def __init__(self, units, actf=keras.activations.swish, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(units, int) or units <= 0:
            raise ValueError("`units` must be a positive integer.")
        self.units = units
        self.actf = get_activation(actf)

    def build(self, input_shape):
        # input_shape: (batch, dim)
        if len(input_shape) != 2:
            raise ValueError("TimeMLPLayer expects rank-2 tensor (batch, dim); got {}".format(input_shape))
        self.dense1 = keras.layers.Dense(
            self.units,
            activation=self.actf,
            kernel_initializer=kernel_init(1.0),
            name="dense1",
        )
        self.dense2 = keras.layers.Dense(
            self.units,
            activation=None,
            kernel_initializer=kernel_init(1.0),
            name="dense2",
        )
        super().build(input_shape)

    def call(self, inputs):
        if not hasattr(inputs, "shape"):
            raise ValueError("Input must be a tensor.")
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "actf": keras.activations.serialize(self.actf),
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        # Deserialize activation if serialized
        actf = config.get("actf", None)
        if isinstance(actf, dict):
            config["actf"] = keras.activations.deserialize(actf)
        return cls(**config)


# This is a custom Keras Layer version of ResidualBlock.
# It allows for more flexibility in using it as a standalone layer in Keras models.
# It can be used in a Keras Sequential model or Functional API.
# It also supports serialization and deserialization with get_config() and from_config().
# It is similar to the original ResidualBlock function but encapsulated in a Keras Layer
# for better integration with Keras workflows.
class ResidualBlock(keras.layers.Layer):
    """
    Custom Keras Layer version of ResidualBlock.
    Args:
        out_channels (int): Number of output channels.
        attention (bool): Whether to use attention.
        num_heads (int): Number of attention heads.
        groups (int): Number of groups for GroupNormalization.
        actf (callable): Activation function.
        dropout_rate (float): Dropout rate.
        kernel_size (int): Convolution kernel size.
        use_cross_attention (bool): Whether to use cross-attention.
    Input: (x, t)
        x: 4D tensor (batch, height, width, channels)
        t: time embedding tensor
    Output:
        Output tensor after residual block and optional attention.
    
    diagram:
    (x, t)  # x: feature map, t: time embedding
    |
    |-----------------------------.
    |                             |
    |                         [If input_width != width]
    |                             |
    |                         1x1 Conv2D
    |                             |
    |------------------------> residual
    |
    |-- GroupNorm --> Activation --> Conv2D (kernel_size)
    |                             |
    |                        [Process t:]
    |                        Activation --> Dense --> Reshape (1,1,-1)
    |                             |
    |------------------------> Add (x + temb)
    |
    |-- GroupNorm --> Activation --> [Dropout if needed] --> Conv2D (kernel_size)
    |                             |
    |------------------------> Add (x + residual)
    |
    |-- [If attention:]
    |     |-- GroupNorm
    |     |-- [If cross-attention:]
    |     |     Broadcast temb to x shape
    |     |     MultiHeadAttention(x, temb)
    |     |-- [Else:]
    |     |     MultiHeadAttention(x, x)
    |     |-- Add (attention + res_output)
    |
    v
    output
    """
    def __init__(
        self, 
        out_channels: int, 
        attention: bool, 
        num_heads: int, 
        groups: int, 
        actf: callable=keras.activations.swish,
        dropout_rate: float=0.0, 
        kernel_size: int=3, 
        use_cross_attention: bool=False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.attention = attention
        self.num_heads = num_heads
        self.groups = groups
        self.actf = actf
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.use_cross_attention = use_cross_attention
        
        # Create layers in __init__ (best practice for Keras)
        self.skip_conv = keras.layers.Conv2D(filters=self.out_channels, kernel_size=1, kernel_initializer=kernel_init(1.0))
        self.temb_act = keras.layers.Activation(self.actf)
        self.temb_dense = keras.layers.Dense(self.out_channels, kernel_initializer=kernel_init(1.0))
        self.temb_reshape = keras.layers.Reshape([1, 1, self.out_channels])
        self.norm1 = keras.layers.GroupNormalization(groups=self.groups)
        self.act1 = keras.layers.Activation(self.actf)
        self.conv1 = keras.layers.Conv2D(self.out_channels, kernel_size=self.kernel_size, padding="same", kernel_initializer=kernel_init(1.0))
        self.dropout = keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.norm2 = keras.layers.GroupNormalization(groups=self.groups)
        self.act2 = keras.layers.Activation(self.actf)
        self.conv2 = keras.layers.Conv2D(self.out_channels, kernel_size=self.kernel_size, padding="same", kernel_initializer=kernel_init(0.0))
        
        if self.attention:
            self.norm_attn = keras.layers.GroupNormalization(groups=self.groups)
            if self.out_channels % self.num_heads != 0:
                raise ValueError("out_channels must be divisible by num_heads for MultiHeadAttention.")
            key_dim = self.out_channels // self.num_heads
            self.mha = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=key_dim, attention_axes=(1, 2))

        if self.attention and self.use_cross_attention:
            self.cross_attn_proj = keras.layers.Dense(self.out_channels)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 2):
            raise ValueError("inputs must be a tuple/list of (x, t)")
        x, t = inputs
        if len(x.shape) != 4:
            raise ValueError("x must be a 4D tensor (batch, height, width, channels)")
        channels_in = x.shape[-1]
        residual = self.skip_conv(x) if channels_in != self.out_channels else x
        temb = self.temb_act(t)
        temb = self.temb_dense(temb)
        temb = self.temb_reshape(temb)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = x + temb
        x = self.norm2(x)
        x = self.act2(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = x + residual # Residual connection
        if self.attention:
            res_output = x
            x = self.norm_attn(x)
            if self.use_cross_attention:
                # Project time embedding to cross-attention context
                ctx = self.cross_attn_proj(t)  # [B, width]
                ctx = ctx[:, None, None, :]  # [B, 1, 1, width] for cross-attention
                x = self.mha(x, ctx, training=training)  # Query from x, Key/Value from ctx
            else:
                x = self.mha(x, x, training=training)
            x = x + res_output
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "attention": self.attention,
            "num_heads": self.num_heads,
            "groups": self.groups,
            "actf": keras.activations.serialize(self.actf),
            "dropout_rate": self.dropout_rate,
            "kernel_size": self.kernel_size,
            "use_cross_attention": self.use_cross_attention
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Deserialize activation if serialized
        actf = config.get("actf", None)
        if isinstance(actf, dict):
            config["actf"] = keras.activations.deserialize(actf)
        return cls(**config)


class SpaceToDepth(keras.layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(block_size, int) or block_size <= 1:
            raise ValueError("`block_size` must be an integer > 1")
        self.block_size = block_size
        
    def call(self, inputs):
        return tf.nn.space_to_depth(inputs, block_size=self.block_size)
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"block_size": self.block_size})
        return cfg
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DepthToSpace(keras.layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(block_size, int) or block_size <= 1:
            raise ValueError("`block_size` must be an integer > 1")
        self.block_size = block_size
        
    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.block_size)
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"block_size": self.block_size})
        return cfg
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DownSample(keras.layers.Layer):
    """Configurable downsampling layer with several strategies.

    Modes:
        conv: Single strided Conv2D (default; matches previous behavior).
        avg:  AveragePool2D (stride=2) followed by 1x1 Conv2D to set channels.
        max:  MaxPool2D (stride=2) followed by 1x1 Conv2D.

    Args:
        width (int): Output channels after downsampling.
        mode (str): One of {'conv','avg','max'}.
        activation (str|callable|None): Optional activation after projection conv.
        kernel_size (int): Kernel size for conv).
    """
    def __init__(self, width, mode="conv", activation=None, kernel_size=3, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if not isinstance(width, int) or width <= 0:
            raise ValueError("`width` must be positive int")
        self.width = width
        self.mode = mode
        self.kernel_size = kernel_size
        self.activation = get_activation(activation) if activation is not None else None

        if mode not in {"conv", "avg", "max", "pixel_unshuffle", "blur_conv"}:
            raise ValueError(f"Unsupported downsample mode: {mode}")

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Input must be 4D (batch, h, w, c)")
        channels_in = input_shape[-1]

        if self.mode == "conv":
            self.op = keras.layers.Conv2D(self.width, kernel_size=self.kernel_size, strides=2, padding="same", kernel_initializer=kernel_init(1.0), name="ds_conv")
        elif self.mode in ("avg", "max"):
            pool_cls = keras.layers.AveragePooling2D if self.mode == "avg" else keras.layers.MaxPooling2D
            self.pool = pool_cls(pool_size=2, strides=2, padding="same", name="ds_pool")
            self.proj = keras.layers.Conv2D(self.width, kernel_size=1, padding="same", kernel_initializer=kernel_init(1.0), name="proj_conv")
        if self.activation is not None:
            self.act_layer = keras.layers.Activation(self.activation)
        else:
            self.act_layer = None
        super().build(input_shape)

    def call(self, inputs):
        if self.mode == "conv":
            x = self.op(inputs)
        elif self.mode in ("avg", "max"):
            x = self.pool(inputs)
            x = self.proj(x)
        else:
            raise RuntimeError("Invalid mode encountered in DownSampleLayer call")
        if self.act_layer is not None:
            x = self.act_layer(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "width": self.width,
            "mode": self.mode,
            "kernel_size": self.kernel_size,
            "activation": keras.activations.serialize(self.activation) if self.activation is not None else None,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        act = config.get("activation", None)
        if isinstance(act, dict):
            config["activation"] = keras.activations.deserialize(act)
        return cls(**config)
        

class UpSample(keras.layers.Layer):
    """Configurable upsampling layer with several strategies.

    Modes:
        resize_conv (default): Interpolation (nearest/bilinear/etc.) then 3x3 Conv2D.
        transposed: Conv2DTranspose with stride=2.
        resize_only: Pure interpolation (no convolution) optionally followed by 1x1 projection if width != in_channels.

    Args:
        width (int): Output channels after upsampling.
        mode (str): One of {"resize_conv","transposed","resize_only"}.
        interpolation (str): Interpolation mode for resize-based methods.
        activation (str|callable|None): Optional activation after main projection.
        kernel_size (int): Kernel size for conv / transposed conv.
    """
    def __init__(self, width, mode="resize_conv", interpolation="nearest", activation=None,
                 kernel_size=3, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if not isinstance(width, int) or width <= 0:
            raise ValueError("`width` must be positive int")
        self.width = width
        self.mode = mode
        self.interpolation = interpolation
        self.activation = get_activation(activation) if activation is not None else None
        self.kernel_size = kernel_size
        if mode not in {"resize_conv", "transposed", "resize_only"}:
            raise ValueError(f"Unsupported upsample mode: {mode}")

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Input must be 4D (batch,h,w,c)")
        in_ch = input_shape[-1]
        if self.mode == "resize_conv":
            self.resize = keras.layers.UpSampling2D(size=2, interpolation=self.interpolation, name="upsample")
            self.conv = keras.layers.Conv2D(self.width, kernel_size=self.kernel_size, padding="same", kernel_initializer=kernel_init(1.0), name="conv")
        elif self.mode == "transposed":
            self.convT = keras.layers.Conv2DTranspose(self.width, kernel_size=self.kernel_size, strides=2, padding="same", kernel_initializer=kernel_init(1.0), name="convT")
        elif self.mode == "resize_only":
            self.resize = keras.layers.UpSampling2D(size=2, interpolation=self.interpolation, name="upsample")
            self.proj = None
            if in_ch != self.width:
                self.proj = keras.layers.Conv2D(self.width, kernel_size=1, padding="same", kernel_initializer=kernel_init(1.0), name="proj")
        if self.activation is not None:
            self.act_layer = keras.layers.Activation(self.activation)
        else:
            self.act_layer = None
        super().build(input_shape)

    def call(self, inputs):
        if self.mode == "resize_conv":
            x = self.resize(inputs)
            x = self.conv(x)
        elif self.mode == "transposed":
            x = self.convT(inputs)
        elif self.mode == "resize_only":
            x = self.resize(inputs)
            if self.proj is not None:
                x = self.proj(x)
        else:
            raise RuntimeError("Invalid mode in UpSample call")
        if self.act_layer is not None:
            x = self.act_layer(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "width": self.width,
            "mode": self.mode,
            "interpolation": self.interpolation,
            "kernel_size": self.kernel_size,
            "activation": keras.activations.serialize(self.activation) if self.activation is not None else None,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        act = config.get("activation", None)
        if isinstance(act, dict):
            config["activation"] = keras.activations.deserialize(act)
        return cls(**config)


class FeatureFusion(keras.layers.Layer):
    """Feature fusion / conditioning layer with multiple strategies.

    Supported fusion_type values:
        concat : Concatenate along channel axis.
        add    : Elementwise add (requires same spatial & channel shapes).
        adain  : Adaptive Instance Normalization (style modulation).
        film   : FiLM-style (Feature-wise Linear Modulation) scale/shift without per-instance norm.

    Inputs:
        x : 4D feature map (B,H,W,Cx)
        condition : Either 2D (B,Cc) or 4D (B,1,1,Cc) or broadcastable to (B,1,1,Cc).

    Args:
        fusion_type (str): One of {"concat","add","adain","film"}.
        epsilon (float): Small constant for numerical stability in normalization.
        use_bias (bool): Whether to use bias term in FiLM/adain transforms.
        projection (bool): If True and fusion_type == concat, apply 1x1 conv to bring channels back to original size.
        activation (str|callable|None): Optional activation after fusion & projection.
    """
    def __init__(self, fusion_type="concat", epsilon=1e-8, use_bias=True,
                 projection=False, activation=None, **kwargs):
        super().__init__(**kwargs)
        if fusion_type not in {"concat", "add", "adain", "film"}:
            raise ValueError("Unsupported fusion_type: {}".format(fusion_type))
        self.fusion_type = fusion_type
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.projection = projection
        self.activation = get_activation(activation) if activation is not None else None

    def build(self, input_shape):
        # input_shape = [x_shape, cond_shape]
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("FeatureFusion expects two inputs: (x, condition)")
        x_shape, cond_shape = input_shape
        if len(x_shape) != 4:
            raise ValueError("x must be 4D (B,H,W,C), got {}".format(x_shape))
        feature_dim = x_shape[-1]

        # Determine condition dimensionality
        if len(cond_shape) == 2:
            cond_dim = cond_shape[-1]
        elif len(cond_shape) == 4:
            cond_dim = cond_shape[-1]
        else:
            raise ValueError("condition must be rank 2 or 4, got {}".format(cond_shape))

        if self.fusion_type in {"adain", "film"}:
            self.scale_transform = keras.layers.Dense(
                feature_dim,
                kernel_initializer=kernel_init(1.0),
                use_bias=self.use_bias,
                name="scale_dense",
            )
            self.bias_transform = keras.layers.Dense(
                feature_dim,
                kernel_initializer=kernel_init(0.0),
                use_bias=self.use_bias,
                name="bias_dense",
            )
        else:
            self.scale_transform = None
            self.bias_transform = None

        if self.fusion_type == "concat" and self.projection:
            self.proj_conv = keras.layers.Conv2D(feature_dim, 1, padding="same", kernel_initializer=kernel_init(1.0), name="proj_conv")
        else:
            self.proj_conv = None

        if self.activation is not None:
            self.act_layer = keras.layers.Activation(self.activation)
        else:
            self.act_layer = None
        super().build(input_shape)

    def _prepare_condition(self, condition):
        # Convert 2D (B,C) to (B,1,1,C)
        if len(condition.shape) == 2:
            condition = condition[:, None, None, :]
        elif len(condition.shape) == 4:
            # If spatial dims are 1x1 we accept as-is
            if not (condition.shape[1] == 1 and condition.shape[2] == 1):
                # Global average pool if spatial condition provided
                condition = tf.reduce_mean(condition, axis=[1, 2], keepdims=True)
        else:
            raise ValueError("Unsupported condition rank: {}".format(condition.shape))
        return condition

    def call(self, inputs, training=None):
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 2):
            raise ValueError("FeatureFusion expects (x, condition)")
        x, condition = inputs
        if len(x.shape) != 4:
            raise ValueError("x must be 4D (B,H,W,C)")

        if self.fusion_type == "concat":
            out = keras.layers.Concatenate(axis=-1)([x, condition])
            if self.proj_conv is not None:
                out = self.proj_conv(out)
        elif self.fusion_type == "add":
            if x.shape != condition.shape:
                raise ValueError("For add fusion, shapes must match: {} vs {}".format(x.shape, condition.shape))
            out = keras.layers.Add()([x, condition])
        else:
            # Prepare condition for modulation
            cond_proc = self._prepare_condition(condition)
            scale = self.scale_transform(cond_proc)
            bias = self.bias_transform(cond_proc)
            if self.fusion_type == "adain":
                mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
                x_norm = (x - mean) / tf.sqrt(var + self.epsilon)
                out = x_norm * scale + bias
            else:  # film
                out = x * (scale) + bias

        if self.act_layer is not None:
            out = self.act_layer(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "fusion_type": self.fusion_type,
            "epsilon": self.epsilon,
            "use_bias": self.use_bias,
            "projection": self.projection,
            "activation": keras.activations.serialize(self.activation) if self.activation is not None else None,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        act = config.get("activation", None)
        if isinstance(act, dict):
            config["activation"] = keras.activations.deserialize(act)
        return cls(**config)

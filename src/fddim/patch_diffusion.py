"""Coordinate helpers for fixed-size Patch Diffusion training and sampling."""

import tensorflow as tf


def normalize_spatial_size(size, name="size", allow_none=False):
    """Normalize an integer or ``(height, width)`` value to a 2-tuple."""
    if size is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be provided")
    if isinstance(size, int) and not isinstance(size, bool):
        if size < 1:
            raise ValueError(f"{name} must contain positive integers")
        return (size, size)
    if (
        isinstance(size, (list, tuple))
        and len(size) == 2
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in size
        )
    ):
        return (size[0], size[1])
    raise ValueError(
        f"{name} must be a positive integer or a two-item (height, width) sequence"
    )


def coordinate_grid(height, width, batch_size=1, dtype=tf.float32):
    """Return normalized absolute ``(x, y)`` coordinates for an image canvas.

    Coordinates follow the Patch Diffusion convention: pixel zero maps to -1
    and the final pixel maps to +1. Singleton dimensions map to -1.
    """
    height = tf.cast(height, tf.int32)
    width = tf.cast(width, tf.int32)
    batch_size = tf.cast(batch_size, tf.int32)

    x = tf.cast(tf.range(width), dtype)
    y = tf.cast(tf.range(height), dtype)
    x_denominator = tf.cast(tf.maximum(width - 1, 1), dtype)
    y_denominator = tf.cast(tf.maximum(height - 1, 1), dtype)
    x = 2.0 * (x / x_denominator - 0.5)
    y = 2.0 * (y / y_denominator - 0.5)
    x_grid, y_grid = tf.meshgrid(x, y, indexing="xy")
    grid = tf.stack((x_grid, y_grid), axis=-1)
    return tf.broadcast_to(
        grid[None, ...],
        tf.stack((batch_size, height, width, tf.constant(2, tf.int32))),
    )


def append_coordinate_channels(images):
    """Append a full-image normalized coordinate grid to ``images``."""
    shape = tf.shape(images)
    coordinates = coordinate_grid(
        shape[1], shape[2], batch_size=shape[0], dtype=images.dtype
    )
    return tf.concat((images, coordinates), axis=-1)


def patch_coordinate_grid(
    height,
    width,
    row_origin,
    column_origin,
    patch_size,
    dtype=tf.float32,
):
    """Return absolute ``(x, y)`` coordinates for one image patch."""
    height = tf.cast(height, tf.int32)
    width = tf.cast(width, tf.int32)
    row_origin = tf.cast(row_origin, tf.int32)
    column_origin = tf.cast(column_origin, tf.int32)
    if isinstance(patch_size, (list, tuple)):
        patch_height = tf.cast(patch_size[0], tf.int32)
        patch_width = tf.cast(patch_size[1], tf.int32)
    else:
        patch_height = tf.cast(patch_size, tf.int32)
        patch_width = patch_height

    columns = column_origin + tf.range(patch_width, dtype=tf.int32)
    rows = row_origin + tf.range(patch_height, dtype=tf.int32)
    x = tf.cast(columns, dtype) / tf.cast(tf.maximum(width - 1, 1), dtype)
    y = tf.cast(rows, dtype) / tf.cast(tf.maximum(height - 1, 1), dtype)
    x = 2.0 * (x - 0.5)
    y = 2.0 * (y - 0.5)
    x_grid, y_grid = tf.meshgrid(x, y, indexing="xy")
    return tf.stack((x_grid, y_grid), axis=-1)

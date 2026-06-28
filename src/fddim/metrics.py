"""Model-free image metrics for generated and real image batches."""

import gzip
import json
from typing import Any, Optional

import numpy as np


PAIRWISE_METRIC_FIELDS = [
    "pairwise_ssim",
    "pairwise_ssim_std",
    "pairwise_ssim_min",
    "pairwise_ssim_max",
    "pairwise_l2",
    "pairwise_l2_std",
    "pairwise_l2_min",
    "pairwise_l2_max",
]

GZIP_RAW_METRIC_FIELDS = [
    "gzip_raw_compression_ratio",
    "gzip_raw_compression_ratio_std",
    "gzip_raw_compression_ratio_min",
    "gzip_raw_compression_ratio_max",
    "gzip_raw_bits_per_value",
    "gzip_raw_bits_per_value_std",
    "gzip_raw_bits_per_value_min",
    "gzip_raw_bits_per_value_max",
    "gzip_raw_bytes_per_image",
    "gzip_compressed_bytes_per_image",
]


def min_images_for_pair_count(num_pairs: int) -> int:
    """Return the smallest image count that can provide num_pairs unique pairs."""
    return int(np.ceil((1.0 + np.sqrt(1.0 + 8.0 * num_pairs)) / 2.0))


def sample_pair_indices(num_images: int, num_pairs: int, seed: int) -> Optional[np.ndarray]:
    """Sample unique unordered image index pairs without replacement."""
    if num_images < 2:
        return None
    max_pairs = num_images * (num_images - 1) // 2
    num_pairs = min(num_pairs, max_pairs)
    all_pairs = np.array(
        [(i, j) for i in range(num_images) for j in range(i + 1, num_images)],
        dtype=np.int32,
    )
    rng = np.random.default_rng(seed)
    selected = rng.choice(all_pairs.shape[0], size=num_pairs, replace=False)
    return all_pairs[selected]


def global_pairwise_ssim(batch_a: np.ndarray, batch_b: np.ndarray) -> np.ndarray:
    """Compute global SSIM for arbitrary-channel image batches in [0, 1]."""
    axes = (1, 2, 3)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_a = np.mean(batch_a, axis=axes)
    mu_b = np.mean(batch_b, axis=axes)
    var_a = np.mean((batch_a - mu_a[:, None, None, None]) ** 2, axis=axes)
    var_b = np.mean((batch_b - mu_b[:, None, None, None]) ** 2, axis=axes)
    cov_ab = np.mean(
        (batch_a - mu_a[:, None, None, None]) * (batch_b - mu_b[:, None, None, None]),
        axis=axes,
    )

    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return numerator / np.maximum(denominator, 1.0e-12)


def pairwise_l2(batch_a: np.ndarray, batch_b: np.ndarray) -> np.ndarray:
    """Compute root mean squared pixel distance for each image pair."""
    diff = batch_a - batch_b
    return np.sqrt(np.mean(diff * diff, axis=(1, 2, 3)))


def compute_pairwise_image_metrics(
    images: np.ndarray,
    num_pairs: int = 1000,
    pair_seed: int = 0,
    duplicate_ssim_threshold: Optional[float] = None,
    duplicate_l2_threshold: Optional[float] = None,
) -> dict[str, Any]:
    """Compute pairwise SSIM/L2 summaries for images in [0, 1]."""
    pairs = sample_pair_indices(images.shape[0], num_pairs, seed=pair_seed)
    if pairs is None:
        return {"num_pairs": 0}

    ssim_values = global_pairwise_ssim(images[pairs[:, 0]], images[pairs[:, 1]])
    l2_values = pairwise_l2(images[pairs[:, 0]], images[pairs[:, 1]])
    metrics: dict[str, Any] = {
        "num_pairs": int(pairs.shape[0]),
        "pairwise_ssim": float(np.mean(ssim_values)),
        "pairwise_ssim_std": float(np.std(ssim_values)),
        "pairwise_ssim_min": float(np.min(ssim_values)),
        "pairwise_ssim_max": float(np.max(ssim_values)),
        "pairwise_l2": float(np.mean(l2_values)),
        "pairwise_l2_std": float(np.std(l2_values)),
        "pairwise_l2_min": float(np.min(l2_values)),
        "pairwise_l2_max": float(np.max(l2_values)),
    }

    if duplicate_ssim_threshold is not None:
        near_duplicate_ssim = ssim_values >= duplicate_ssim_threshold
        metrics["near_duplicate_ssim_threshold"] = float(duplicate_ssim_threshold)
        metrics["near_duplicate_rate_ssim"] = float(np.mean(near_duplicate_ssim))
    else:
        near_duplicate_ssim = None

    if duplicate_l2_threshold is not None:
        near_duplicate_l2 = l2_values <= duplicate_l2_threshold
        metrics["near_duplicate_l2_threshold"] = float(duplicate_l2_threshold)
        metrics["near_duplicate_rate_l2"] = float(np.mean(near_duplicate_l2))
    else:
        near_duplicate_l2 = None

    if near_duplicate_ssim is not None and near_duplicate_l2 is not None:
        metrics["near_duplicate_rate_any"] = float(
            np.mean(near_duplicate_ssim | near_duplicate_l2)
        )

    return metrics


def compute_reference_image_metrics(
    images: np.ndarray,
    num_pairs: int = 1000,
    pair_seed: int = 0,
    source: str = "real_train_images",
) -> dict[str, Any]:
    """Compute reference pairwise metrics for a real image batch in [0, 1]."""
    if images.shape[0] < 2:
        raise ValueError("Need at least 2 images to compute reference metrics.")

    metrics = {
        "source": source,
        "value_range": "[0, 1]",
        "num_images": int(images.shape[0]),
        "image_height": int(images.shape[1]),
        "image_width": int(images.shape[2]),
        "image_channels": int(images.shape[3]),
        "pair_seed": int(pair_seed),
    }
    metrics.update(
        compute_pairwise_image_metrics(
            images,
            num_pairs=num_pairs,
            pair_seed=pair_seed,
        )
    )
    return metrics


def gzip_raw_metrics(images: np.ndarray) -> dict[str, float]:
    """Compute per-sample gzip metrics from arbitrary-channel images in [0, 1]."""
    images_u8 = np.clip(images * 255.0, 0.0, 255.0).astype(np.uint8)
    compression_ratios = []
    bits_per_value = []
    raw_bytes_per_image = []
    compressed_bytes_per_image = []
    for image in images_u8:
        raw = np.ascontiguousarray(image).tobytes()
        compressed = gzip.compress(raw)
        raw_len = len(raw)
        compressed_len = len(compressed)
        num_values = image.size
        compression_ratios.append(raw_len / compressed_len)
        bits_per_value.append(8.0 * compressed_len / num_values)
        raw_bytes_per_image.append(raw_len)
        compressed_bytes_per_image.append(compressed_len)

    compression_ratios = np.asarray(compression_ratios, dtype=np.float64)
    bits_per_value = np.asarray(bits_per_value, dtype=np.float64)
    raw_bytes_per_image = np.asarray(raw_bytes_per_image, dtype=np.float64)
    compressed_bytes_per_image = np.asarray(compressed_bytes_per_image, dtype=np.float64)
    return {
        "gzip_raw_compression_ratio": float(np.mean(compression_ratios)),
        "gzip_raw_compression_ratio_std": float(np.std(compression_ratios)),
        "gzip_raw_compression_ratio_min": float(np.min(compression_ratios)),
        "gzip_raw_compression_ratio_max": float(np.max(compression_ratios)),
        "gzip_raw_bits_per_value": float(np.mean(bits_per_value)),
        "gzip_raw_bits_per_value_std": float(np.std(bits_per_value)),
        "gzip_raw_bits_per_value_min": float(np.min(bits_per_value)),
        "gzip_raw_bits_per_value_max": float(np.max(bits_per_value)),
        "gzip_raw_bytes_per_image": float(np.mean(raw_bytes_per_image)),
        "gzip_compressed_bytes_per_image": float(np.mean(compressed_bytes_per_image)),
    }


def format_metric_value(value: Any, digits: int = 3, as_string: bool = False) -> Any:
    """Format floats to a fixed number of significant digits."""
    if value is None:
        return "" if as_string else value
    if isinstance(value, (float, np.floating)):
        text = f"{value:.{digits}g}"
        return text if as_string else float(text)
    return value


def format_metric_values(
    metrics: dict[str, Any],
    digits: int = 3,
    as_strings: bool = False,
) -> dict[str, Any]:
    return {
        key: format_metric_value(value, digits=digits, as_string=as_strings)
        for key, value in metrics.items()
    }


def save_metrics_json(path: str, metrics: dict[str, Any], digits: int = 3) -> None:
    formatted = format_metric_values(metrics, digits=digits, as_strings=False)
    with open(path, "w") as f:
        json.dump(formatted, f, indent=2)
        f.write("\n")

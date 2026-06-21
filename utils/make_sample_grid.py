#!/usr/bin/env python3
"""Create a sample image grid from generated PNG/JPEG files."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def parse_color(value: str) -> tuple[int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise argparse.ArgumentTypeError("Color must be a 6-digit hex value, for example ffffff.")
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color must be a valid 6-digit hex value.") from exc


def collect_image_paths(input_dir: Path) -> list[Path]:
    paths = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=natural_key)


def select_image_paths(
    paths: list[Path],
    count: int,
    *,
    start: int,
    shuffle: bool,
    seed: int,
) -> list[Path]:
    if shuffle:
        rng = random.Random(seed)
        paths = list(paths)
        rng.shuffle(paths)
        selected = paths[:count]
    else:
        selected = paths[start : start + count]
    if len(selected) < count:
        raise ValueError(f"Need {count} images, but only found {len(selected)} matching images.")
    return selected


def load_tile(path: Path, tile_size: int | None) -> Image.Image:
    with Image.open(path) as image:
        tile = image.convert("RGB")
        if tile_size is not None:
            tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
        return tile.copy()


def make_grid(
    image_paths: list[Path],
    *,
    rows: int,
    cols: int,
    tile_size: int | None,
    padding: int,
    background: tuple[int, int, int],
) -> Image.Image:
    first_tile = load_tile(image_paths[0], tile_size)
    tile_w, tile_h = first_tile.size
    grid_w = cols * tile_w + (cols + 1) * padding
    grid_h = rows * tile_h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), background)

    for index, path in enumerate(image_paths):
        row = index // cols
        col = index % cols
        tile = first_tile if index == 0 else load_tile(path, tile_size)
        if tile.size != (tile_w, tile_h):
            tile = tile.resize((tile_w, tile_h), Image.Resampling.NEAREST)
        x = padding + col * (tile_w + padding)
        y = padding + row * (tile_h + padding)
        grid.paste(tile, (x, y))
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PNG sample grid from a generated image folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Generated image directory. PNG/JPEG files are searched recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the sample grid.",
    )
    parser.add_argument("--rows", type=int, default=10, help="Number of grid rows.")
    parser.add_argument("--cols", type=int, default=10, help="Number of grid columns.")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Optional square tile size. Defaults to each image's native size.",
    )
    parser.add_argument("--padding", type=int, default=2, help="Padding in pixels.")
    parser.add_argument(
        "--background",
        type=parse_color,
        default=parse_color("ffffff"),
        help="Background hex color, for example ffffff or 111111.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset in naturally sorted image paths. Ignored with --shuffle.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly sample images instead of using naturally sorted order.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive.")
    if args.tile_size is not None and args.tile_size <= 0:
        raise ValueError("--tile-size must be positive when provided.")
    if args.padding < 0:
        raise ValueError("--padding must be non-negative.")
    if args.start < 0:
        raise ValueError("--start must be non-negative.")

    input_dir = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    count = args.rows * args.cols
    paths = collect_image_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No PNG/JPEG images found under {input_dir}")
    selected = select_image_paths(
        paths,
        count,
        start=args.start,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    grid = make_grid(
        selected,
        rows=args.rows,
        cols=args.cols,
        tile_size=args.tile_size,
        padding=args.padding,
        background=args.background,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    print(f"Wrote {count} images to {output}")
    print(f"Source: {input_dir}")


if __name__ == "__main__":
    main()

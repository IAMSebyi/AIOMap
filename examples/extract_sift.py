from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rich
import tyro

from aiomap.core.images import ImageCollection, ImageLoadOptions
from aiomap.features.extractors.points.sift import SIFTFeatureExtractor
from aiomap.visualization.features import save_keypoint_visualization


CacheType = Literal["none", "preload", "lazy_lru"]


@dataclass
class Args:
    """Run SIFT feature extraction on an image folder."""

    images: Path
    """Path to the input image directory."""

    output: Path
    """Path to the output directory."""

    max_images: int = -1
    """Maximum number of images to process. -1 means all images."""

    max_size: int = -1
    """Resize images so the longest side is at most this value. -1 disables resizing."""

    cache_type: CacheType = "lazy_lru"
    """Image cache policy."""

    max_cached_images: int = 32
    """Maximum number of images kept in memory. -1 means unlimited."""

    num_workers: int = 0
    """Number of workers for image preloading. 0 or 1 means serial."""

    device: Literal["auto", "cpu", "cuda"] = "auto"
    """Device used by pycolmap SIFT."""

    debug_plots: bool = True
    """Whether to save keypoint visualizations."""

    max_debug_images: int = 8
    """Maximum number of debug visualizations to save."""

    max_debug_keypoints: int = 1000
    """Maximum number of keypoints drawn per debug image."""


def main(args: Args) -> None:
    rich.print("[bold cyan]AIOMap SIFT extraction example[/bold cyan]")

    images = ImageCollection(
        root=args.images,
        options=ImageLoadOptions(
            color_mode="grayscale",
            dtype="uint8",
            max_size=args.max_size,
        ),
        cache_type=args.cache_type,
        max_cached_images=args.max_cached_images,
        num_workers=args.num_workers,
    )

    if args.max_images > 0:
        selected_names = images.names[: args.max_images]
    else:
        selected_names = images.names

    rich.print(f"[green]Found[/green] {len(images)} images.")
    rich.print(f"[green]Processing[/green] {len(selected_names)} images.")
    rich.print(f"[green]Output directory:[/green] {args.output}")

    # For now, create a lightweight view by temporarily limiting iteration.
    # This avoids changing ImageCollection just for the example script.
    original_names = images.names
    images.names = selected_names

    extractor = SIFTFeatureExtractor(device=args.device)

    result = extractor.run(
        images=images,
        output_dir=args.output,
    )

    images.names = original_names

    rich.print("[bold green]SIFT extraction finished.[/bold green]")
    rich.print(f"Total keypoints: {result.metadata['total_keypoints']}")
    rich.print(f"Average keypoints/image: {result.metadata['avg_keypoints_per_image']:.2f}")

    artifact_path = result.metadata.get("artifact_path")
    if artifact_path:
        rich.print(f"Feature artifact: [bold]{artifact_path}[/bold]")

    if args.debug_plots:
        debug_dir = args.output / "debug_keypoints"
        debug_names = selected_names[: args.max_debug_images]

        rich.print(f"[cyan]Saving debug plots to:[/cyan] {debug_dir}")

        for image_name in debug_names:
            image_data = images.get_by_name(image_name)
            keypoints, _ = result.features[image_name]

            output_path = debug_dir / f"{Path(image_name).stem}_keypoints.jpg"

            save_keypoint_visualization(
                image_data=image_data,
                keypoints=keypoints,
                output_path=output_path,
                max_keypoints=args.max_debug_keypoints,
            )

        rich.print(f"[bold green]Saved {len(debug_names)} debug visualizations.[/bold green]")


if __name__ == "__main__":
    main(tyro.cli(Args))

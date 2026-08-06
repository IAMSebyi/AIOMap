from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from os import cpu_count
from typing import Iterator, Literal, Optional, Tuple, TypeAlias

import cv2
import numpy as np
import numpy.typing as npt


IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}


ColorMode: TypeAlias = Literal['rgb', 'grayscale']
ImageDType: TypeAlias = Literal['uint8', 'float32']
ImageInterpolation: TypeAlias = Literal['area', 'linear', 'cubic']

CacheType: TypeAlias = Literal['none', 'preload', 'lazy_lru']


@dataclass(frozen=True)
class ImageLoadOptions:
    """Image loading and preprocessing options."""

    color_mode: ColorMode = "rgb"
    """Output color mode."""
    dtype: ImageDType = "uint8"
    """Output image dtype. float32 images are scaled to [0, 1]."""
    max_size: int = -1
    """Resize so the longest side is at most max_size. -1 means no max-size resize."""
    fixed_size: Optional[Tuple[int, int]] = None
    """Optional fixed output size as (width, height). Overrides max_size when set."""
    interpolation: ImageInterpolation = "area"
    """Interpolation mode used for resizing."""


@dataclass
class ImageData:
    """Loaded image data and metadata."""

    name: str
    """Image name relative to the collection root."""
    path: Path
    """Absolute or collection-root image path."""
    array: npt.NDArray[np.generic]
    """Loaded image array."""
    width: int
    """Current image width."""
    height: int
    """Current image height."""
    original_width: int
    """Original image width before resizing."""
    original_height: int
    """Original image height before resizing."""
    scale_x: float
    """Scale from original width to current width: current_width / original_width."""
    scale_y: float
    """Scale from original height to current height: current_height / original_height."""
    color_mode: ColorMode
    """Output color mode."""
    dtype: ImageDType
    """Output dtype."""


def _cv2_color_mode(color_mode: ColorMode) -> int:
    if color_mode == "rgb":
        return cv2.IMREAD_COLOR
    if color_mode == "grayscale":
        return cv2.IMREAD_GRAYSCALE
    raise ValueError(f"Unsupported color mode: {color_mode}")


def _cv2_interpolation(interpolation: ImageInterpolation) -> int:
    if interpolation == "area":
        return cv2.INTER_AREA
    if interpolation == "linear":
        return cv2.INTER_LINEAR
    if interpolation == "cubic":
        return cv2.INTER_CUBIC
    raise ValueError(f"Unsupported interpolation mode: {interpolation}")


def load_image(path: Path, options: ImageLoadOptions) -> ImageData:
    """Load and preprocess a single file image"""

    # Read image
    image = cv2.imread(str(path), _cv2_color_mode(options.color_mode))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if options.color_mode == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    original_height, original_width = image.shape[:2]
    new_width, new_height = original_width, original_height

    if options.fixed_size is not None:
        new_width, new_height = options.fixed_size
    elif options.max_size is not None and options.max_size > 0:
        longest_side = max(original_width, original_height)
        if longest_side > options.max_size:
            scale = options.max_size / longest_side
            new_width = int(round(original_width * scale))
            new_height = int(round(original_height * scale))

    if (new_width, new_height) != (original_width, original_height):
        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=_cv2_interpolation(options.interpolation),
        )

    # Assign image data type and proper value range
    if options.dtype == "float32":
        # Assume [0...1] range
        image = image.astype(np.float32) / 255.0
    elif options.dtype == "uint8":
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image dtype: {options.dtype}")

    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]

    return ImageData(
        name=path.name,
        path=path,
        array=image,
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
        scale_x=width / original_width,
        scale_y=height / original_height,
        color_mode=options.color_mode,
        dtype=options.dtype,
    )


class ImageCollection:
    """Collection of images with optional in-memory caching."""

    def __init__(
        self,
        root: Path,
        options: ImageLoadOptions,
        cache_type: CacheType = "none",
        max_cached_images: int = -1,
        num_workers: Optional[int] = None,
    ):
        # Check if a proper directory has been parsed
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError("Images directory is not a valid directory")

        # Search for images in root directory and construct the list of sorted image paths
        self.paths = sorted(
            file for file in self.root.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_FORMATS
        )
        self.names = [file.name for file in self.paths]
        self.name_to_path = dict(zip(self.names, self.paths))

        # Check if any images were found in the directory
        if not self.paths:
            raise FileNotFoundError("No images found in the specified directory")

        # Initialize caching and image storage
        self.options = options
        self.cache_type = cache_type
        self.max_cached_images = max_cached_images
        self.cache: OrderedDict[str, ImageData] = OrderedDict()

        # Max cached images must be -1 (unlimited) or >= 0
        if self.max_cached_images < -1:
            raise ValueError("max_cached_images must be -1 (unlimited) or a positive integer")

        # If max_cached_images is 0, disable caching
        if self.cache_type != "none" and self.max_cached_images == 0:
            self.cache_type = "none"

        # Preload images if cache_type is "preload"
        if self.cache_type == "preload":
            self._preload(num_workers=num_workers)

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def _load_by_name(self, name: str) -> ImageData:
        """Load by name helper function for multithreading."""
        return load_image(self.name_to_path[name], self.options)

    def _preload(self, num_workers: Optional[int]) -> None:
        """Preload images into memory using multiple threads."""
        if self.max_cached_images == -1:
            names_to_load = self.names
        else:
            names_to_load = self.names[: self.max_cached_images]
        workers = max(1, min(8, cpu_count() or 1)) if num_workers is None else num_workers

        if workers <= 1:
            for name in names_to_load:
                self.cache[name] = self._load_by_name(name)
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for name, image_data in zip(
                names_to_load,
                executor.map(self._load_by_name, names_to_load),
            ):
                self.cache[name] = image_data

    def _evict_if_needed(self) -> None:
        """Evict least recently used images if the cache exceeds max_cached_images."""
        if self.max_cached_images == -1:
            return

        while len(self.cache) > self.max_cached_images:
            self.cache.popitem(last=False)

    def get_by_name(self, name: str) -> ImageData:
        """Get image by filename. Loads the image if not already cached."""
        if name not in self.name_to_path:
            raise KeyError(f"Unknown image name: {name}")

        if self.cache_type != "none" and name in self.cache:
            image_data = self.cache.pop(name)
            self.cache[name] = image_data
            return image_data

        image_data = self._load_by_name(name)

        if self.cache_type == "lazy_lru":
            self.cache[name] = image_data
            self._evict_if_needed()

        return image_data

    def get_by_index(self, index: int) -> ImageData:
        """Get image by index. Loads the image if not already cached."""

        return self.get_by_name(self.names[index])
    
    def get_path_by_name(self, name: str) -> Path:
        """Get image path by filename."""
        if name not in self.name_to_path:
            raise KeyError(f"Unknown image name: {name}")
        return self.name_to_path[name]

    def clear_cache(self) -> None:
        """Clear all images stored in memory."""
        self.cache.clear()

    def cached_count(self) -> int:
        """Return the number of images currently cached in memory."""
        return len(self.cache)

    def is_cached(self, name: str) -> bool:
        """Check if an image is currently cached in memory."""
        return name in self.cache
    
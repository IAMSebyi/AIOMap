from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt

from aiomap.core.images import ImageData


def draw_keypoints(
    image_data: ImageData,
    keypoints: npt.NDArray[np.float32],
    max_keypoints: int = 1000,
    radius: int = 2,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> npt.NDArray[np.uint8]:
    """Draw keypoints on an image.

    Assumes keypoints are stored in original image coordinates.
    """

    if image_data.array.ndim == 2:
        vis = cv2.cvtColor(image_data.array, cv2.COLOR_GRAY2RGB)  # type: ignore
    else:
        vis = image_data.array.copy()

    if vis.dtype != np.uint8:
        vis = np.clip(vis * 255.0, 0, 255).astype(np.uint8)  # type: ignore

    points = keypoints[:, :2].astype(np.float32, copy=True)

    # Convert from original-image coordinates to current loaded-image coordinates.
    points[:, 0] *= image_data.width / image_data.original_width
    points[:, 1] *= image_data.height / image_data.original_height

    if max_keypoints > 0 and len(points) > max_keypoints:
        indices = np.linspace(0, len(points) - 1, max_keypoints).astype(np.int64)
        points = points[indices]

    for x, y in points:
        cv2.circle(
            vis,  # type: ignore
            (int(round(x)), int(round(y))),
            radius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    return vis  # type: ignore


def save_keypoint_visualization(
    image_data: ImageData,
    keypoints: npt.NDArray[np.float32],
    output_path: Path,
    max_keypoints: int = 1000,
    radius: int = 2,
) -> None:
    """Save keypoint visualization as an image file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis = draw_keypoints(
        image_data=image_data,
        keypoints=keypoints,
        max_keypoints=max_keypoints,
        radius=radius,
    )

    # OpenCV writes BGR; vis is RGB.
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
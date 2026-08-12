from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple

import numpy as np
import numpy.typing as npt

from aiomap.core.images import ImageData

@dataclass
class FeatureExtractionResult:
    """Feature extraction result"""

    features: Dict[str, Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = field(default_factory=dict)
    """Dictionary mapping image names to tuples of (keypoints, descriptors)"""
    feature_type: Literal['points', 'lines', 'dense'] = 'points'
    """Type of features extracted"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata dictionary containing additional information about the feature extraction process"""


def rescale_keypoints(keypoints: npt.NDArray[np.float32], image_data: ImageData) -> npt.NDArray[np.float32]:
    """Rescale keypoints to original image size"""
    if image_data.array.shape != (image_data.original_height, image_data.original_width):
        scale_x = image_data.original_width / image_data.width
        scale_y = image_data.original_height / image_data.height
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

    return keypoints

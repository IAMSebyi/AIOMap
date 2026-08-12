from pathlib import Path

import numpy as np
import pycolmap

from aiomap.common.utils import resolve_device
from aiomap.core.features import FeatureExtractionResult, rescale_keypoints
from aiomap.core.images import ImageCollection
from aiomap.core.types import Device
from aiomap.features.extractors.base_extractor import FeatureExtractor

class SIFTFeatureExtractor(FeatureExtractor):
    """SIFT feature extractor"""

    def __init__(
        self,
        max_image_size: int = -1,
        num_threads: int = -1,
        gpu_index: str = "-1",
        max_num_features: int = 8192, 
        first_octave: int = -1, 
        num_octaves: int = 4, 
        octave_resolution: int = 3, 
        peak_threshold: float = 0.006666666666666667, 
        edge_threshold: float = 10.0, 
        estimate_affine_shape: bool = False, 
        max_num_orientations: int = 2, 
        upright: bool = False, 
        darkness_adaptivity: bool = False, 
        domain_size_pooling: bool = False, 
        dsp_min_scale: float = 0.16666666666666666, 
        dsp_max_scale: float = 3.0, 
        dsp_num_scales: int = 10, 
        normalization: pycolmap.Normalization = pycolmap.Normalization.L1_ROOT,
        device: Device = 'auto' 
    ):
        device = resolve_device(device=device)
        
        # Initialize pycolmap SIFT extractor
        self.options = pycolmap.FeatureExtractionOptions(
            type=pycolmap.FeatureExtractorType.SIFT,
            max_image_size=max_image_size,
            num_threads=num_threads,
            use_gpu=(device == 'cuda'),
            gpu_index=gpu_index,
            sift=pycolmap.SiftExtractionOptions(
                max_num_features=max_num_features,
                first_octave=first_octave,
                num_octaves=num_octaves,
                octave_resolution=octave_resolution,
                peak_threshold=peak_threshold,
                edge_threshold=edge_threshold,
                estimate_affine_shape=estimate_affine_shape,
                max_num_orientations=max_num_orientations,
                upright=upright,
                darkness_adaptivity=darkness_adaptivity,
                domain_size_pooling=domain_size_pooling,
                dsp_min_scale=dsp_min_scale,
                dsp_max_scale=dsp_max_scale,
                dsp_num_scales=dsp_num_scales,
                normalization=normalization
            )
        )

        self.sift = pycolmap.Sift(options=self.options)

    def run(self, images: ImageCollection, output_dir: Path) -> FeatureExtractionResult:
        # Check if images are grayscale uint8, as required by SIFT
        if images.options.color_mode != "grayscale" or images.options.dtype != "uint8":
            raise ValueError("SIFTFeatureExtractor expects grayscale uint8 images.")
        
        # Create output directory if it does not exist (pipeline.py already creates the output directory, 
        # but we create it here as well to ensure that the extractor can be run independently)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract features sequentially using pycolmap
        features = {}

        for image_name in images:
            image_data = images.get_by_name(image_name)

            keypoints, descriptors = self.sift.extract(image=image_data.array)

            # Rescale keypoints to original image size
            keypoints = rescale_keypoints(keypoints, image_data)
            
            features[image_name] = (
                keypoints.astype(np.float32, copy=False),
                descriptors.astype(np.float32, copy=False),
            )

        # Save features to output directory
        return FeatureExtractionResult(
            features=features,
            feature_type='points',
            metadata={
                'extractor': 'SIFT',
                'options': self.options.todict(),
                'num_images': len(images),
                'avg_keypoints_per_image':
                    np.mean([len(kp) for kp, _ in features.values()]) 
                    if features else 0,
                'total_keypoints': sum(len(kp) for kp, _ in features.values()),
                'output_dir': str(output_dir)
            }
        )

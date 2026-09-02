from abc import ABC, abstractmethod
from pathlib import Path

from aiomap.core.features import FeatureExtractionResult
from aiomap.core.images import ImageCollection


class FeatureExtractor(ABC):
    """Base feature extractor class"""

    @abstractmethod
    def run(self, images: ImageCollection, output_dir: Path) -> FeatureExtractionResult:
        raise NotImplementedError

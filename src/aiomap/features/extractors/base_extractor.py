from abc import ABC, abstractmethod
from pathlib import Path

from aiomap.core.types import ImageCollection, FeatureExtractionResult


class FeatureExtractor(ABC):
    """Base feature extractor class"""

    @abstractmethod
    def run(self, images: ImageCollection, output_dir: Path) -> FeatureExtractionResult:
        raise NotImplementedError

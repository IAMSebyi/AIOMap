from abc import ABC, abstractmethod
from pathlib import Path

from aiomap.core.features import FeatureExtractionResult
from aiomap.core.images import ImageCollection
from aiomap.core.types import (
    FeatureMatchingResult,
    MappingResult
)


class Mapper(ABC):
    """Base mapper class"""

    @abstractmethod
    def run(
        self, 
        images: ImageCollection, 
        features: FeatureExtractionResult,
        matches: FeatureMatchingResult,
        output_dir: Path
    ) -> MappingResult:
        raise NotImplementedError

from abc import ABC, abstractmethod
from pathlib import Path

from aiomap.core.types import (
    ImageCollection, 
    FeatureExtractionResult, 
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

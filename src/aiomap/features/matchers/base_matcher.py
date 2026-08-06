from abc import ABC, abstractmethod
from pathlib import Path

from aiomap.core.images import ImageCollection
from aiomap.core.types import (
    FeatureExtractionResult, 
    PairSelectionResult,
    FeatureMatchingResult
)


class FeatureMatcher(ABC):
    """Base feature matcher class"""

    @abstractmethod
    def run(
        self, 
        images: ImageCollection, 
        pairs: PairSelectionResult, 
        features: FeatureExtractionResult,
        output_dir: Path
    ) -> FeatureMatchingResult:
        raise NotImplementedError

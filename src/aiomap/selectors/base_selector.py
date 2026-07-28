from abc import ABC, abstractmethod
from pathlib import Path

from aiomap.core.types import (
    ImageCollection, 
    PairSelectionResult
)


class ImagePairSelector(ABC):
    """Base image pair selector class"""

    @abstractmethod
    def run(
        self, 
        images: ImageCollection, 
        output_dir: Path
    ) -> PairSelectionResult:
        raise NotImplementedError

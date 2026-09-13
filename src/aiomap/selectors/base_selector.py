from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

from aiomap.core.images import ImageCollection
from aiomap.core.types import PairSelectionResult


class ImagePairSelector(ABC):
    """Base image pair selector class"""

    @staticmethod
    def _write_txt(pairs: List[Tuple[str, str]], output: Path) -> None:
        """Write pairs to a text file"""
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(f"{pair[0]} {pair[1]}\n")
        
    @abstractmethod
    def run(
        self, 
        images: ImageCollection, 
        output_dir: Path
    ) -> PairSelectionResult:
        raise NotImplementedError

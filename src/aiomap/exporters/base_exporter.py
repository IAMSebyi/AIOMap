from abc import ABC, abstractmethod
from pathlib import Path

from aiomap.core.types import (
    MappingResult,
    ExportResult
)


class Exporter(ABC):
    """Base exporter class"""

    @abstractmethod
    def run(self, mapping: MappingResult, output_dir: Path) -> ExportResult:
        raise NotImplementedError

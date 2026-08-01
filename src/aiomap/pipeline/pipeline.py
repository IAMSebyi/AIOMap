from pathlib import Path
from typing import Optional

from aiomap.core.types import ImageCollection, PipelineResult
from aiomap.exporters.base_exporter import Exporter
from aiomap.features.extractors.base_extractor import FeatureExtractor
from aiomap.features.matchers.base_matcher import FeatureMatcher
from aiomap.mappers.base_mapper import Mapper
from aiomap.selectors.base_selector import ImagePairSelector


class Pipeline:
    """
    Pipeline runner
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        matcher: FeatureMatcher,
        selector: ImagePairSelector,
        mapper: Mapper,
        exporter: Optional[Exporter] = None
    ):
        self.extractor = extractor
        self.matcher = matcher
        self.selector = selector
        self.mapper = mapper
        self.exporter = exporter

    def run(self, images_dir: Path, output_dir: Path) -> PipelineResult:
        # Get absolute paths; using Path() in case parsed directory paths are strings
        images_dir = Path(images_dir).resolve()
        output_dir = Path(output_dir).resolve()

        # Build image collection from images directory
        images = ImageCollection(images_dir=images_dir)

        # Create output directory (and its parents) if it does not exist; otherwise check if it's a valid directory
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        elif not output_dir.is_dir():
            raise NotADirectoryError("Output directory exists, but is not a valid directory.")

        # Create subdirectories for different pipeline stages inside output directory
        artifacts_dir = output_dir / 'artifacts'
        mapping_dir = output_dir / 'mapping'
        artifacts_dir.mkdir(exist_ok=True)
        mapping_dir.mkdir(exist_ok=True)

        # Extract features
        features = self.extractor.run(images=images, output_dir=artifacts_dir)

        # Select image pairs for matching
        pairs = self.selector.run(images=images, output_dir=artifacts_dir)

        # Match features
        matches = self.matcher.run(images=images, pairs=pairs, features=features, output_dir=artifacts_dir)

        # Run mapping
        mapping = self.mapper.run(images=images, features=features, matches=matches, output_dir=mapping_dir)

        # Export to a certain format if requested
        export = None
        if self.exporter:
            # Create export subdirectory inside output directory
            export_dir = output_dir / 'export'
            export_dir.mkdir(exist_ok=True)

            export = self.exporter.run(mapping=mapping, output_dir=export_dir)

        return PipelineResult(
            features=features,
            pairs=pairs,
            matches=matches,
            mapping=mapping,
            export=export
        )

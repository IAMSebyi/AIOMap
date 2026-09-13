from pathlib import Path
from typing import List, Tuple

from aiomap.core.images import ImageCollection
from aiomap.core.types import PairSelectionResult
from aiomap.selectors.base_selector import ImagePairSelector


class ExhaustiveImagePairSelector(ImagePairSelector):
    """Exhaustive image pair selector."""

    def __init__(
        self,
        bidirectional: bool = False,
    ):
        self.bidirectional = bidirectional

    def run(
        self,
        images: ImageCollection,
        output_dir: Path,
    ) -> PairSelectionResult:
        # Create output directory if it does not exist (pipeline.py already creates the output directory,
        # but we create it here as well to ensure that the selector can be run independently)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pairs list and image count
        pairs: List[Tuple[str, str]] = []
        image_count = len(images)

        # Add all unique image pairs
        for idx1 in range(image_count):
            for idx2 in range(idx1 + 1, image_count):
                image_name1 = images.names[idx1]
                image_name2 = images.names[idx2]

                pairs.append((image_name1, image_name2))

                if self.bidirectional:
                    pairs.append((image_name2, image_name1))

        # Write pairs to a text file
        pairs_path = output_dir / "pairs.txt"
        self._write_txt(pairs, pairs_path)

        return PairSelectionResult(
            pairs=pairs,
            metadata={
                "selector": "exhaustive",
                "num_images": image_count,
                "num_pairs": len(pairs),
                "bidirectional": self.bidirectional,
                "pairs_path": str(pairs_path),
            },
        )

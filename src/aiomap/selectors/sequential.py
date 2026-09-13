from pathlib import Path
from typing import List, Optional

from aiomap.core.images import ImageCollection
from aiomap.core.types import PairSelectionResult
from aiomap.selectors.base_selector import ImagePairSelector


class SequentialImagePairSelector(ImagePairSelector):
    """Sequential image pair selector"""

    def __init__(
        self,
        num_neighbors: int = 10,
        min_gap: int = 1,
        max_gap: Optional[int] = None,
        stride: int = 1,
        bidirectional: bool = False,
        loop_closure_every: Optional[int] = None,
        loop_closure_neighbors: int = 1,
        wrap_around: bool = False,
    ):
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be greater than or equal to 1.")

        if min_gap < 1:
            raise ValueError("min_gap must be greater than or equal to 1.")

        if stride < 1:
            raise ValueError("stride must be greater than or equal to 1.")

        if max_gap is not None and max_gap < min_gap:
            raise ValueError("max_gap must be greater than or equal to min_gap.")

        if loop_closure_every is not None and loop_closure_every <= 0:
            raise ValueError("loop_closure_every must be greater than 0 when set.")

        if loop_closure_neighbors <= 0:
            raise ValueError("loop_closure_neighbors must be greater than 0.")

        self.num_neighbors = num_neighbors
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.stride = stride
        self.bidirectional = bidirectional
        self.loop_closure_every = loop_closure_every
        self.loop_closure_neighbors = loop_closure_neighbors
        self.wrap_around = wrap_around

    def _generate_sequential_offsets(self) -> List[int]:
        offsets = []

        for i in range(self.num_neighbors):
            offset = (i + 1) * self.stride

            if self.max_gap is not None and offset > self.max_gap:
                break
            if offset >= self.min_gap:
                offsets.append(offset)

        return offsets

    def _generate_loop_closure_offsets(self) -> List[int]:
        offsets = []

        if self.loop_closure_every:
            for i in range(self.loop_closure_neighbors):
                offsets.append((i + 1) * self.loop_closure_every)

        return offsets

    def run(
        self,
        images: ImageCollection, 
        output_dir: Path
    ) -> PairSelectionResult:
        # Create output directory if it does not exist (pipeline.py already creates the output directory, 
        # but we create it here as well to ensure that the extractor can be run independently)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pairs set and image count
        pairs_set = set()
        image_count = len(images)

        # Generate sequential offsets and loop closure offsets, if they exist
        offsets = self._generate_sequential_offsets()
        offsets.extend(self._generate_loop_closure_offsets())

        # Add pairs to set
        for source in range(image_count):
            for offset in offsets:
                neighbor = source + offset

                # If neighbor index is larger than the image count, either skip or wrap around if requested
                if neighbor >= image_count:
                    if self.wrap_around:
                        neighbor = neighbor % image_count
                    else:
                        continue

                # Skip if neighbor is the same as source
                if neighbor == source:
                    continue

                # Add pair to set, ensuring that the smaller index is first if not bidirectional
                if self.bidirectional:
                    pairs_set.add((source, neighbor))
                    pairs_set.add((neighbor, source))
                else:
                    pairs_set.add((min(source, neighbor), max(source, neighbor)))

        # Build names pairs list from the indices pairs set
        pairs_indices = sorted(pairs_set)
        pairs = [(images.names[idx1], images.names[idx2]) for idx1, idx2 in pairs_indices]

        # Write pairs to a text file
        pairs_path = output_dir / "pairs.txt"
        self._write_txt(pairs, pairs_path)

        return PairSelectionResult(
            pairs=pairs,
            metadata={
                "selector": "exhaustive",
                "num_images": image_count,
                "num_pairs": len(pairs),
                "num_neighbors": self.num_neighbors,
                "min_gap": self.min_gap,
                "max_gap": self.max_gap,
                "stride": self.stride,
                "bidirectional": self.bidirectional,
                "loop_closure_every": self.loop_closure_every,
                "loop_closure_neighbors": self.loop_closure_neighbors,
                "wrap_around": self.wrap_around,
                "pairs_path": str(pairs_path),
            }
        )

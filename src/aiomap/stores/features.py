from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import zarr


FeatureArray = npt.NDArray[np.float32]
FeatureTuple = Tuple[FeatureArray, FeatureArray]
AccessMode = Literal["r", "r+", "w", "w-", "a"]


def _safe_image_key(image_name: str) -> str:
    """Convert an image name to a safe Zarr group key.

    Zarr uses "/" as a group separator, so image names that may contain
    subdirectories need to be normalized before being used as group names.
    """
    return image_name.replace("/", "__").replace("\\", "__")


class ZarrFeatureStore:
    """Zarr-backed store for local image features.

    Current layout:

        features.zarr/
            attrs:
                type: "features"
                format_version: 1
                feature_type: "points"
                extractor: optional str

            images/
                <safe_image_key>/
                    attrs:
                        image_name: original image name
                        optional metadata fields
                    keypoints:   [N, K] float32
                    descriptors: [N, D] float32

    The store intentionally keeps a minimal interface so extractors and
    matchers do not need to know Zarr-specific details.
    """

    def __init__(
        self,
        path: Path,
        mode: AccessMode = "a",
        feature_type: str = "points",
        extractor: Optional[str] = None,
    ):
        self.path = Path(path)

        self.root = zarr.open_group(str(self.path), mode=mode)
        self.root.attrs["type"] = "features"
        self.root.attrs["format_version"] = 1
        self.root.attrs["feature_type"] = feature_type

        if extractor is not None:
            self.root.attrs["extractor"] = extractor

        self.images_group = self.root.require_group("images")

    def write(
        self,
        image_name: str,
        keypoints: npt.NDArray,
        descriptors: npt.NDArray,
        overwrite: bool = True,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """Write features for one image."""

        key = _safe_image_key(image_name)

        if key in self.images_group:
            if not overwrite:
                raise FileExistsError(
                    f"Features already exist for image: {image_name}"
                )
            del self.images_group[key]

        group = self.images_group.create_group(key)

        group.attrs["image_name"] = image_name

        if metadata:
            for metadata_key, value in metadata.items():
                if self._is_supported_attr_value(value):
                    group.attrs[metadata_key] = value  # type: ignore

        group.create_array(
            "keypoints",
            data=np.asarray(keypoints, dtype=np.float32),
        )

        group.create_array(
            "descriptors",
            data=np.asarray(descriptors, dtype=np.float32),
        )

    def read(self, image_name: str) -> FeatureTuple:
        """Read features for one image."""

        key = _safe_image_key(image_name)

        if key not in self.images_group:
            raise KeyError(f"No features found for image: {image_name}")

        group = self.images_group[key]

        keypoints = np.asarray(group["keypoints"][:], dtype=np.float32)  # type: ignore
        descriptors = np.asarray(group["descriptors"][:], dtype=np.float32)  # type: ignore

        return keypoints, descriptors

    def read_all(self) -> Dict[str, FeatureTuple]:
        """Read all stored features into memory."""

        return {image_name: self.read(image_name) for image_name in self.names()}

    def exists(self, image_name: str) -> bool:
        """Check whether features exist for an image."""

        return _safe_image_key(image_name) in self.images_group

    def names(self) -> List[str]:
        """Return stored image names."""

        image_names = []

        for key in self.images_group.keys():
            image_names.append(self.images_group[key].attrs["image_name"])

        return sorted(image_names)

    def metadata(self, image_name: str) -> Dict[str, object]:
        """Return metadata stored for one image."""

        key = _safe_image_key(image_name)

        if key not in self.images_group:
            raise KeyError(f"No features found for image: {image_name}")

        group = self.images_group[key]

        return {
            key: value
            for key, value in group.attrs.items()
            if key != "image_name"
        }

    def __len__(self) -> int:
        return len(self.images_group)

    @staticmethod
    def _is_supported_attr_value(value: object) -> bool:
        """Return whether a value can be safely stored as a Zarr attribute."""

        return isinstance(value, (str, int, float, bool)) or value is None

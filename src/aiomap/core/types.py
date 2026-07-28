from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


IMAGE_FORMATS = [
    '.jpg',
    '.jpeg',
    '.png',
    '.tif',
    '.tiff',
    '.bmp',
    '.gif',
    '.heic',
]


class ImageCollection:
    """Collection of images"""

    root: Path
    """Images root directory path"""
    paths: List[Path]
    """Sorted list of image paths"""
    names: List[str]
    """Sorted list of image names"""

    def __init__(self, images_dir: Path):
        # Check if a proper directory has been parsed
        if not images_dir.is_dir():
            raise NotADirectoryError("Images directory is not a valid directory")

        # Search for images in root directory and construct the list of image paths
        self.root = images_dir
        self.paths = []
        self.names = []

        for file in self.root.iterdir():
            if file.suffix.lower() in IMAGE_FORMATS:
                self.paths.append(file)

        # Check if any images were found in the directory
        if not self.paths:
            raise FileNotFoundError("No images found in the specified directory")

        # Sort the list of image paths and construct the list of image names
        self.paths.sort()
        self.names = [file.name for file in self.paths]


@dataclass
class FeatureExtractionResult:
    """Feature extraction result"""

    features_path: Optional[Path] = None
    """Path to file containing the extracted features"""
    database_path: Optional[Path] = None
    """Path to COLMAP database file containing the extracted features"""
    feature_type: Literal['points', 'lines', 'dense'] = 'points'
    """Type of features extracted"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata dictionary containing additional information about the feature extraction process"""


@dataclass
class PairSelectionResult:
    """Pair selection result"""

    pairs: List[Tuple[str, str]] = field(default_factory=list)
    """List of selected image pairs"""
    pairs_path: Optional[Path] = None
    """Path to file containing the selected image pairs"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata dictionary containing additional information about the pair selection process"""


@dataclass
class FeatureMatchingResult:
    """Feature matching result"""

    matches_path: Optional[Path] = None
    """Path to file containing the matched features"""
    database_path: Optional[Path] = None
    """Path to COLMAP database file containing the matched features"""
    feature_type: Literal['points', 'lines', 'dense'] = 'points'
    """Type of features matched"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata dictionary containing additional information about the feature matching process"""


@dataclass
class MappingResult:
    """Mapping result"""

    reconstruction_path: Optional[Path] = None
    """Path to directory containing the mapping results"""
    reconstruction_type: Literal['points', 'lines', 'other'] = 'points'
    """Type of reconstruction"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata dictionary containing additional information about the mapping process"""


@dataclass
class ExportResult:
    """Export result"""

    export_path: Optional[Path] = None
    """Path to directory containing the exported results"""
    export_format: Literal['colmap', 'nerfstudio', 'other'] = 'colmap'
    """Format of the exported results"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata dictionary containing additional information about the export process"""
    
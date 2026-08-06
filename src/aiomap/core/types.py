from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias


Device: TypeAlias = Literal['auto', 'cpu', 'cuda']


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


@dataclass
class PipelineResult:
    """Pipeline runner result"""

    features: FeatureExtractionResult
    """Feature extraction result"""
    pairs: PairSelectionResult
    """Image pair selection result"""
    matches: FeatureMatchingResult
    """Feature matching result"""
    mapping: MappingResult
    """Mapping result"""
    export: Optional[ExportResult]
    """Optional export result"""

# AIOMap: All-In-One Mapping

AIOMap is an early-stage open-source project for building modular Structure-from-Motion and 3D mapping pipelines.

The long-term goal is to make sparse reconstruction workflows easier to configure, run, inspect, compare, and extend by splitting them into interchangeable components such as image pair selection, feature extraction, feature matching, mapping, and model export.

## Status

AIOMap is currently in early development.

The first milestone is to implement a minimal pycolmap-based SfM pipeline using:

- SIFT feature extraction
- basic image pair selection
- SIFT feature matching
- COLMAP-style sparse mapping

At this stage, the repository only contains the minimal project foundation. The core package structure and pipeline implementation will be added incrementally.

## Vision

Most 3D reconstruction tools expose powerful pipelines, but they are often difficult to customize, compare, or combine across different feature extractors, matchers, and mapping backends.

AIOMap aims to become a developer-friendly workflow engine for experimenting with modular 3D reconstruction pipelines.

In the long term, the project aims to support:

- classical SfM pipelines based on COLMAP / pycolmap
- learned local features and matchers
- configurable image pair selection
- sparse reconstruction diagnostics
- exporters for common 3D reconstruction and neural rendering formats
- future integration with point-based, line-based, and dense correspondence pipelines

## License

AIOMap is released under the MIT License.

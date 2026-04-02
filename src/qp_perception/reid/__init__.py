"""Re-ID subpackage: feature extraction and appearance gallery."""

from .extractor import ReIDConfig, ReIDExtractor
from .gallery import AppearanceGallery, GalleryConfig

# BPU backend is S100P-only — import lazily
# from .bpu_extractor import BPUReIDConfig, BPUReIDExtractor

__all__ = [
    "ReIDExtractor",
    "ReIDConfig",
    "AppearanceGallery",
    "GalleryConfig",
]

"""
Re-ID Feature Extractor.
=========================

Extracts appearance feature vectors from image crops for person
re-identification.  Supports two backbones:

    - **osnet_x1_0** (default): OSNet trained on MSMT17 person Re-ID dataset
      (4101 identities).  512-dim features, ~2.2M params.  Dramatically better
      at distinguishing different people compared to generic classifiers.
      Weights auto-downloaded from HuggingFace on first use.

    - **mobilenet**: MobileNetV3-Small with ImageNet weights.  576-dim features,
      ~2.5M params.  Fast but NOT trained for person discrimination — only
      suitable as a fallback when OSNet weights are unavailable.

The extractor is stateless — it only transforms image crops to feature vectors.
Matching and gallery management belong to AppearanceGallery (separation of concerns).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2 as _cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class ReIDConfig:
    """Configuration for the Re-ID feature extractor.

    Attributes
    ----------
    backbone : str
        "osnet_x1_0" (person Re-ID, recommended) or "mobilenet" (fallback).
    crop_width : int
        Crop resize width in pixels.
    crop_height : int
        Crop resize height in pixels.  128x256 approximates person aspect ratio.
    device : str
        PyTorch device.  "" or "auto" → auto-detect.
    batch_size : int
        Maximum batch size for inference.
    """

    backbone: str = "osnet_x1_0"
    crop_width: int = 128
    crop_height: int = 256
    device: str = ""
    batch_size: int = 32


class ReIDExtractor:
    """Extract L2-normalized appearance features from bounding-box crops.

    Usage::

        extractor = ReIDExtractor()
        features = extractor.extract(frame, bboxes)
        # features: np.ndarray, shape (N, feature_dim), L2-normalized rows
    """

    def __init__(self, config: ReIDConfig | None = None) -> None:
        cfg = config or ReIDConfig()
        self._crop_w = cfg.crop_width
        self._crop_h = cfg.crop_height
        self._batch_size = cfg.batch_size
        self._backbone_name = cfg.backbone

        if cfg.device and cfg.device != "auto":
            self._device = torch.device(cfg.device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model, self._FEATURE_DIM = self._build_model(cfg.backbone)
        self._transform = self._build_transform()

        logger.info(
            "ReIDExtractor ready  backbone=%s  dim=%d  device=%s  crop=%dx%d",
            cfg.backbone,
            self._FEATURE_DIM,
            self._device,
            self._crop_w,
            self._crop_h,
        )

    @property
    def feature_dim(self) -> int:
        return self._FEATURE_DIM

    def extract(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> np.ndarray:
        """Extract features for a list of bounding boxes.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).
        bboxes : list of (x, y, w, h)
            Bounding boxes in pixel coordinates.

        Returns
        -------
        np.ndarray
            Shape ``(N, feature_dim)``, L2-normalized feature vectors.
            Returns shape ``(0, feature_dim)`` if bboxes is empty.
        """
        if not bboxes:
            return np.empty((0, self._FEATURE_DIM), dtype=np.float32)

        crops = self._crop_and_preprocess(frame, bboxes)
        if crops is None:
            return np.empty((0, self._FEATURE_DIM), dtype=np.float32)

        features = self._forward(crops)
        return features

    def extract_single(
        self, frame: np.ndarray, x: float, y: float, w: float, h: float
    ) -> np.ndarray:
        """Extract feature for a single bbox. Returns shape ``(feature_dim,)``."""
        result = self.extract(frame, [(x, y, w, h)])
        if len(result) == 0:
            return np.zeros(self._FEATURE_DIM, dtype=np.float32)
        return result[0]

    def _build_model(self, backbone: str) -> tuple[nn.Module, int]:
        """Build the feature extraction backbone.

        Returns (model, feature_dim).
        """
        if backbone.startswith("osnet"):
            from .osnet import build_osnet
            model = build_osnet(variant=backbone, device=self._device)
            return model, model.feature_dim

        # Fallback: MobileNetV3-Small (ImageNet, not Re-ID specific)
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model = nn.Sequential(base.features, base.avgpool, nn.Flatten())
        model.to(self._device)
        model.eval()
        return model, 576

    @staticmethod
    def _build_transform() -> transforms.Compose:
        """ImageNet normalization transform (applied to tensors)."""
        return transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _crop_and_preprocess(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> torch.Tensor | None:
        """Crop, resize, and batch-preprocess bboxes from frame.

        Returns a tensor of shape (N, 3, crop_h, crop_w) or None if all crops fail.
        """
        h_img, w_img = frame.shape[:2]
        tensors: list[torch.Tensor] = []

        for x, y, w, h in bboxes:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w_img, int(x + w))
            y2 = min(h_img, int(y + h))

            if x2 - x1 < 4 or y2 - y1 < 4:
                tensors.append(torch.zeros(3, self._crop_h, self._crop_w))
                continue

            crop = frame[y1:y2, x1:x2]
            crop = _cv2.resize(crop, (self._crop_w, self._crop_h))
            # BGR → RGB, HWC → CHW, uint8 → float32 [0, 1]
            crop_rgb = _cv2.cvtColor(crop, _cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
            tensor = self._transform(tensor)
            tensors.append(tensor)

        if not tensors:
            return None

        return torch.stack(tensors)

    @torch.no_grad()
    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        """Run inference in batches and L2-normalize."""
        all_feats: list[np.ndarray] = []

        for i in range(0, len(batch), self._batch_size):
            chunk = batch[i : i + self._batch_size].to(self._device)
            feats = self._model(chunk)
            feats = feats.cpu().numpy()
            all_feats.append(feats)

        features = np.concatenate(all_feats, axis=0)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        features = features / norms
        return features.astype(np.float32)

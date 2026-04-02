"""BPU-accelerated Re-ID feature extractor for Horizon S100P (Nash BPU).

Replaces PyTorch OSNet inference with BPU .hbm model inference.
~3ms per crop on Nash BPU vs ~15-20ms on CPU PyTorch.

Usage::

    from qp_perception.reid.bpu_extractor import BPUReIDExtractor

    extractor = BPUReIDExtractor("/path/to/osnet_x1_0_128x256.hbm")
    features = extractor.extract(frame, bboxes)  # same API as ReIDExtractor
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default search paths for the .hbm model on S100P
_MODEL_CANDIDATES = [
    "/opt/lingtu/models/osnet_*.hbm",
    "/home/sunrise/data/models/osnet_*.hbm",
    "/opt/nav/models/osnet_*.hbm",
]


@dataclass
class BPUReIDConfig:
    """Configuration for BPU Re-ID extractor.

    Attributes
    ----------
    model_path : str
        Path to the .hbm model. Empty string = auto-discover.
    crop_width : int
        Crop resize width (must match ONNX export).
    crop_height : int
        Crop resize height (must match ONNX export).
    feature_dim : int
        Output feature dimension (512 for OSNet x1.0).
    """

    model_path: str = ""
    crop_width: int = 128
    crop_height: int = 256
    feature_dim: int = 512


class BPUReIDExtractor:
    """BPU-accelerated Re-ID feature extractor.

    Drop-in replacement for ``ReIDExtractor`` on S100P hardware.
    Uses ``HB_HBMRuntime`` for Nash BPU inference.

    The ``extract()`` API is identical to ``ReIDExtractor``:

    - Input: BGR frame + list of (x, y, w, h) bboxes
    - Output: (N, feature_dim) L2-normalized float32 array
    """

    def __init__(self, config: BPUReIDConfig | None = None) -> None:
        cfg = config or BPUReIDConfig()
        self._crop_w = cfg.crop_width
        self._crop_h = cfg.crop_height
        self._FEATURE_DIM = cfg.feature_dim
        self._rt = None
        self._mname = None
        self._load_model(cfg.model_path)

    def _load_model(self, model_path: str) -> None:
        """Load .hbm model via HB_HBMRuntime."""
        import glob as _glob
        from hbm_runtime import HB_HBMRuntime

        path = model_path
        if not path:
            for pattern in _MODEL_CANDIDATES:
                matches = sorted(_glob.glob(pattern))
                if matches:
                    path = matches[-1]
                    break
            if not path:
                raise FileNotFoundError(
                    f"No OSNet .hbm model found. Searched: {_MODEL_CANDIDATES}"
                )

        self._rt = HB_HBMRuntime(path)
        self._mname = self._rt.model_names[0]

        # Warmup inference to verify model works
        dummy = np.zeros((1, 3, self._crop_h, self._crop_w), dtype=np.float32)
        out = self._rt.run({"input": dummy})[self._mname]
        # Find the feature output (should be shape (1, feature_dim))
        for name, arr in out.items():
            if arr.shape[-1] == self._FEATURE_DIM:
                self._output_name = name
                break
        else:
            # Fallback: use first output
            self._output_name = list(out.keys())[0]
            actual_dim = out[self._output_name].shape[-1]
            logger.warning(
                "BPU Re-ID: expected dim=%d, got %d from output '%s'",
                self._FEATURE_DIM, actual_dim, self._output_name,
            )
            self._FEATURE_DIM = actual_dim

        logger.info(
            "BPUReIDExtractor ready  model=%s  dim=%d  crop=%dx%d",
            os.path.basename(path), self._FEATURE_DIM,
            self._crop_w, self._crop_h,
        )

    @property
    def feature_dim(self) -> int:
        return self._FEATURE_DIM

    def extract(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> np.ndarray:
        """Extract L2-normalized features for a list of bounding boxes.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).
        bboxes : list of (x, y, w, h)
            Bounding boxes in pixel coordinates.

        Returns
        -------
        np.ndarray
            Shape (N, feature_dim), L2-normalized.
        """
        if not bboxes:
            return np.empty((0, self._FEATURE_DIM), dtype=np.float32)

        batch = self._crop_and_preprocess(frame, bboxes)
        if batch is None:
            return np.empty((0, self._FEATURE_DIM), dtype=np.float32)

        features = self._forward(batch)
        return features

    def extract_single(
        self, frame: np.ndarray, x: float, y: float, w: float, h: float,
    ) -> np.ndarray:
        """Extract feature for a single bbox. Returns shape (feature_dim,)."""
        result = self.extract(frame, [(x, y, w, h)])
        if len(result) == 0:
            return np.zeros(self._FEATURE_DIM, dtype=np.float32)
        return result[0]

    def _crop_and_preprocess(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> np.ndarray | None:
        """Crop, resize, and preprocess bboxes for BPU inference.

        Returns (N, 3, crop_h, crop_w) float32 array (NCHW, RGB, normalized).
        """
        h_img, w_img = frame.shape[:2]
        crops = []

        for x, y, w, h in bboxes:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w_img, int(x + w))
            y2 = min(h_img, int(y + h))

            if x2 - x1 < 4 or y2 - y1 < 4:
                crops.append(np.zeros((3, self._crop_h, self._crop_w), dtype=np.float32))
                continue

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (self._crop_w, self._crop_h))
            # BGR → RGB, HWC → CHW, uint8 → float32 [0, 1], ImageNet normalize
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop_rgb = (crop_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            crops.append(crop_rgb.transpose(2, 0, 1))  # CHW

        if not crops:
            return None

        return np.stack(crops).astype(np.float32)

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        """Run BPU inference and L2-normalize."""
        # BPU processes one at a time (or small batch depending on model)
        all_feats = []
        for i in range(len(batch)):
            inp = batch[i:i+1]  # (1, 3, H, W)
            out = self._rt.run({"input": inp})[self._mname]
            feat = out[self._output_name]
            all_feats.append(feat.reshape(1, -1))

        features = np.concatenate(all_feats, axis=0)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        features = features / norms
        return features.astype(np.float32)

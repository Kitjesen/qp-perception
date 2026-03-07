"""
YoloDetector: YOLO11n real inference via ultralytics.

Responsibilities (single):
    - Load model once, run inference per frame, output normalized Detection list.
    - Class whitelist filtering and confidence threshold are handled here.
    - No tracking, no coordinate transform -- those belong to downstream modules.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from qp_perception.types import BoundingBox, Detection

logger = logging.getLogger(__name__)


class YoloDetector:
    """
    Wraps ultralytics YOLO for single-responsibility detection.

    Parameters
    ----------
    model_path : str
        Path to a ``.pt`` or ``.onnx`` weight file, or an ultralytics model
        name like ``"yolo11n.pt"`` (will auto-download on first use).
    confidence_threshold : float
        Detections below this score are discarded **before** returning.
    nms_iou_threshold : float
        IoU threshold for Non-Maximum Suppression inside YOLO.
    class_whitelist : optional sequence of class names
        If provided, only these COCO class names are kept (e.g. ``["person", "car"]``).
        ``None`` means keep all classes.
    device : str
        ``"cuda:0"``, ``"cpu"``, or ``""`` for ultralytics auto-select.
    img_size : int
        Input image size for YOLO inference (longer side).
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.45,
        nms_iou_threshold: float = 0.45,
        class_whitelist: Sequence[str] | None = None,
        device: str = "",
        img_size: int = 640,
    ) -> None:
        from ultralytics import YOLO  # type: ignore[import-untyped]

        self._model = YOLO(model_path)
        self._conf = confidence_threshold
        self._iou = nms_iou_threshold
        self._device = device
        self._img_size = img_size

        self._id_to_name: dict[int, str] = self._model.names
        self._allowed_ids: list[int] | None = None
        if class_whitelist is not None:
            name_lower_map = {v.lower(): k for k, v in self._id_to_name.items()}
            self._allowed_ids = [
                name_lower_map[n.lower()] for n in class_whitelist if n.lower() in name_lower_map
            ]
            if not self._allowed_ids:
                logger.warning(
                    "class_whitelist %s matched no COCO classes; all detections will be empty.",
                    class_whitelist,
                )

        logger.info(
            "YoloDetector ready  model=%s  conf=%.2f  iou=%.2f  whitelist=%s  device=%s",
            model_path,
            self._conf,
            self._iou,
            class_whitelist,
            device or "auto",
        )

    def detect(self, frame: object, timestamp: float) -> list[Detection]:
        """Run YOLO inference on a single BGR frame (np.ndarray)."""
        if not isinstance(frame, np.ndarray):
            logger.warning("YoloDetector.detect received non-ndarray frame; returning empty.")
            return []

        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._img_size,
            device=self._device or None,
            classes=self._allowed_ids,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                cls_name = self._id_to_name.get(int(cls_ids[i]), "unknown")
                detections.append(
                    Detection(
                        bbox=BoundingBox(x=float(x1), y=float(y1), w=float(w), h=float(h)),
                        confidence=float(confs[i]),
                        class_id=cls_name,
                        timestamp=timestamp,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

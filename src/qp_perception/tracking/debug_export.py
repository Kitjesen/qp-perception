"""Debug-only export from qp_perception Track objects to supervision.

``supervision`` is an **optional** dependency — this module is never imported
by the runtime tracking pipeline.  An ``ImportError`` is raised only when a
function is actually *called* without supervision installed.

Usage (offline replay / debug tools)::

    from qp_perception.tracking.debug_export import to_sv_detections

    tracks = tracker.detect_and_track(frame, ts)
    detections = to_sv_detections(tracks)
    # render with any sv annotator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import supervision as sv


def to_sv_detections(tracks: Sequence[object]) -> "sv.Detections":
    """Convert Track-like objects into ``sv.Detections``.

    Preserves ``bbox``, ``confidence``, ``tracker_id``, and stores
    ``class_id`` and ``mask_center`` in ``detections.data``.

    Raises:
        ImportError: if ``supervision`` or ``numpy`` is not installed.
    """
    import numpy as np
    import supervision as sv

    count = len(tracks)
    if count == 0:
        return sv.Detections.empty()

    xyxy = np.empty((count, 4), dtype=np.float32)
    confidence = np.empty((count,), dtype=np.float32)
    tracker_id = np.empty((count,), dtype=np.int32)
    class_name: list[str] = []
    mask_center: list[tuple[float, float] | None] = []

    for idx, track in enumerate(tracks):
        bbox = track.bbox
        xyxy[idx] = (
            float(bbox.x),
            float(bbox.y),
            float(bbox.x + bbox.w),
            float(bbox.y + bbox.h),
        )
        confidence[idx] = float(track.confidence)
        tracker_id[idx] = int(track.track_id)
        class_name.append(str(getattr(track, "class_id", "unknown")))

        center = getattr(track, "mask_center", None)
        mask_center.append(
            (float(center[0]), float(center[1])) if center is not None else None
        )

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        tracker_id=tracker_id,
        data={"class_name": class_name, "mask_center": mask_center},
        metadata={"source": "qp_perception"},
    )


def annotate_frame(
    frame: "np.ndarray",
    tracks: Sequence[object],
    *,
    trace_length: int = 30,
) -> "np.ndarray":
    """One-call debug overlay: box + label + trace on *frame*.

    Convenience wrapper for offline replay tools that don't want to set up
    annotators manually.
    """
    import supervision as sv

    detections = to_sv_detections(tracks)
    if len(detections.xyxy) == 0:
        return frame

    annotated = frame.copy()
    labels = [f"ID:{int(tid)}" for tid in detections.tracker_id]
    annotated = sv.TraceAnnotator(
        color_lookup=sv.ColorLookup.TRACK,
        trace_length=trace_length,
        thickness=2,
    ).annotate(scene=annotated, detections=detections)
    annotated = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.TRACK,
    ).annotate(scene=annotated, detections=detections)
    annotated = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.TRACK,
    ).annotate(scene=annotated, detections=detections, labels=labels)
    return annotated

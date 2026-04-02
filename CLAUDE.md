# qp-perception

- **Package name**: qp-perception (import as `qp_perception`)
- **Version**: 0.2.0
- **Origin**: extracted from nova-rws, now standalone library used by LingTu + RWS

## Architecture

Protocol-based dependency injection. Three core interfaces in `interfaces.py`:

- `Detector` — frame → `list[Detection]`
- `Tracker` — detections → `list[Track]` with stable IDs
- `TargetSelector` — tracks → `TargetObservation | None`

## Module Map

```
qp_perception/
├── types.py              BoundingBox, Detection, Track, TrackState, TargetObservation
├── interfaces.py         Detector, Tracker, TargetSelector (Protocol)
├── config.py             DetectorConfig, SelectorConfig, SelectorWeights
├── kalman.py             CentroidKalman2D (CV 4-state), CentroidKalmanCA (CA 6-state)
│
├── detection/
│   ├── yolo.py           YoloDetector (ultralytics)
│   └── passthrough.py    PassthroughDetector (pre-computed detections)
│
├── tracking/
│   ├── fusion.py         FusionMOT — main tracker (see below)
│   ├── iou.py            SimpleIoUTracker (lightweight fallback)
│   ├── yolo_seg.py       YoloSegTracker (segmentation + tracking)
│   └── cmc.py            CameraMotionCompensator (sparse optical flow)
│
├── selection/
│   ├── weighted.py       WeightedTargetSelector (multi-factor scoring)
│   ├── person_following.py  PersonFollowingSelector (lock-on state machine)
│   ├── rotating.py       RotatingTargetSelector (round-robin)
│   └── multi.py          WeightedMultiTargetSelector + TargetAllocator
│
└── reid/                 (torch-dependent, lazy import)
    ├── extractor.py      ReIDExtractor (OSNet x1.0 / MobileNet fallback)
    ├── osnet.py           OSNet architecture + HuggingFace weight loading
    └── gallery.py        AppearanceGallery (dual-prototype EMA + temporal bank)
```

## FusionMOT — Core Tracker

3-stage matching with fused cost matrix:

| Stage | Detections | Tracks | Cost | Purpose |
|-------|-----------|--------|------|---------|
| 1 | High-conf | All active | IoU + Appearance + Motion + Height [+ Skeleton] | Primary matching |
| 2 | Low-conf | Unmatched | IoU only (ByteTrack style) | Recover occluded |
| 3 | Unmatched high | Lost + patience | Wide gate + Re-ID | Track recovery |

**Academic references**: Deep OC-SORT (ICASSP 2023), ByteTrack (ECCV 2022), Hybrid-SORT (AAAI 2024), OC-SORT (CVPR 2023), MOTIP (CVPR 2025).

**Key features**:
- Kalman CA 6-state (position + velocity + acceleration) with adaptive Q
- EMAP: ego-motion aware prediction for gimbal platforms
- Selective Re-ID: `update_selective()` skips extraction for high-IoU matches (~60-70% savings)
- Dynamic Appearance: confidence-gated EMA prevents gallery pollution
- Adaptive Weighting: boosts appearance weight when discriminative
- OCM/ORU: observation-centric momentum corrects Kalman drift after occlusion
- Skeleton tracking: pose-guided matching via bone proportion descriptors

## PersonFollowingSelector

State machine for robot person-following:

```
UNLOCKED → lock_track(id) → LOCKED → target missing → SEARCHING → timeout → LOST
                                ↑          track reappears          ↑
                                └──────────────────────────────────┘
```

- `lock_track(id)`: bind to FusionMOT track_id
- `lock_by_crop_similarity()`: match via Re-ID features
- `auto_lock`: optionally lock highest-confidence person
- `needs_reselect`: True when LOST (trigger VLM re-selection)

## Key Conventions

- Frozen dataclasses for immutable value types
- `Protocol` (typing) for all interfaces — no abstract base classes
- Heavy dependencies (torch) lazy-imported via `qp_perception.reid`
- All public types importable from `qp_perception` top level
- Re-ID classes require `pip install qp-perception[reid]`

## Integration Points

| Consumer | How |
|----------|-----|
| **LingTu PersonTracker** | `enable_fusion_tracking()` → FusionMOT + OSNet + PersonFollowingSelector |
| **LingTu bpu_qp_bridge** | BPUDetector → FusionMOT.update_selective() → WeightedTargetSelector |
| **RWS** | YoloDetector → FusionMOT → WeightedTargetSelector (original use case) |

## Build & Test

```bash
pip install -e ".[dev,reid]"  # editable install with all deps
pytest                         # 90 tests
ruff check src/                # lint
mypy src/                      # type check
```

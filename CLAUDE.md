# qp-perception

- **Package name**: qp-perception (import as `qp_perception`)
- **Origin**: extracted from nova-rws perception module (`products/nova-rws/src/rws_tracking/perception/`)
- **Version**: 0.1.0

## Architecture

Protocol-based dependency injection. Three core interfaces defined in `interfaces.py`:

- `Detector` — takes a frame, returns `list[Detection]`
- `Tracker` — takes detections, returns `list[Track]` with stable IDs
- `TargetSelector` — picks the best target from tracks, returns `TargetObservation | None`

## Submodules

| Submodule | Contents |
|-----------|----------|
| `types` | Frozen dataclasses: BoundingBox, Detection, Track, TargetObservation; TrackState enum |
| `interfaces` | Protocol classes for DI |
| `config` | DetectorConfig, SelectorConfig, SelectorWeights |
| `kalman` | CentroidKalman2D (CV), CentroidKalmanCA (CA) |
| `detection/` | Concrete detector implementations |
| `tracking/` | Concrete tracker implementations |
| `selection/` | Concrete target selector implementations |
| `reid/` | Re-ID feature extraction, appearance gallery (torch-dependent) |

## Key Conventions

- Frozen dataclasses for immutable value types
- `Protocol` (typing) for all interfaces -- no abstract base classes
- All imports use `qp_perception.` prefix (not `rws_tracking.`)
- Heavy dependencies (torch) are lazy-imported

## Testing

```bash
pytest                  # run tests
ruff check src/         # lint
mypy src/               # type check
```

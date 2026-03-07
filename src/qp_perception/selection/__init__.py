"""Selection subpackage: target selection and allocation strategies."""

from .multi import TargetAllocator, TargetAssignment, WeightedMultiTargetSelector
from .rotating import RotatingTargetSelector
from .weighted import WeightedTargetSelector

__all__ = [
    "WeightedTargetSelector",
    "WeightedMultiTargetSelector",
    "TargetAllocator",
    "TargetAssignment",
    "RotatingTargetSelector",
]

"""Selection subpackage: target selection and allocation strategies."""

from .multi import TargetAllocator, TargetAssignment, WeightedMultiTargetSelector
from .person_following import FollowingConfig, PersonFollowingSelector
from .rotating import RotatingTargetSelector
from .weighted import WeightedTargetSelector

__all__ = [
    "WeightedTargetSelector",
    "WeightedMultiTargetSelector",
    "TargetAllocator",
    "TargetAssignment",
    "RotatingTargetSelector",
    "PersonFollowingSelector",
    "FollowingConfig",
]

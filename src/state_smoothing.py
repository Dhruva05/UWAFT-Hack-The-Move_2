"""Temporal smoothing utilities for traffic light state stabilization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional


class StableState(str, Enum):
    NONE = "NONE"
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


_RAW_TO_STABLE = {
    None: StableState.NONE,
    "redlight": StableState.RED,
    "yellowlight": StableState.YELLOW,
    "greenlight": StableState.GREEN,
}


@dataclass
class TemporalStateFilter:
    """
    Debounce raw frame-level detections into a stable state.

    A new state is only accepted when it appears for `min_consecutive`
    consecutive updates.
    """

    window_size: int = 5
    min_consecutive: int = 3

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.min_consecutive < 1:
            raise ValueError("min_consecutive must be >= 1")
        if self.min_consecutive > self.window_size:
            raise ValueError("min_consecutive cannot exceed window_size")

        self._history: Deque[StableState] = deque(maxlen=self.window_size)
        self._stable_state: StableState = StableState.NONE

    @property
    def stable_state(self) -> StableState:
        return self._stable_state

    def reset(self) -> None:
        self._history.clear()
        self._stable_state = StableState.NONE

    def update(self, raw_class: Optional[str]) -> StableState:
        """Push one raw class label and return the current stable state."""
        observed = _RAW_TO_STABLE.get(raw_class, StableState.NONE)
        self._history.append(observed)

        if len(self._history) < self.min_consecutive:
            return self._stable_state

        tail = list(self._history)[-self.min_consecutive :]
        if all(state == tail[0] for state in tail):
            self._stable_state = tail[0]

        return self._stable_state


def stable_state_from_raw(raw_class: Optional[str], smoother: TemporalStateFilter) -> StableState:
    """Convenience helper that applies smoothing to a raw class label."""
    return smoother.update(raw_class)

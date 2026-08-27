"""Realtime frame transport, ball tracks, and table motion state.

The capture thread publishes only its newest frame.  The analysis thread never
queues stale frames, which is more important to aiming latency than preserving
every intermediate animation frame.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from aimtool.vision import Ball

Point = Tuple[float, float]
_UNIQUE_LABELS = {"白球", "黄球", "绿球", "棕球", "蓝球", "粉球", "黑球"}


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    captured_at: float
    frame: np.ndarray
    self_mask: Optional[np.ndarray]


class FrameStore:
    """Thread-safe latest-frame handoff; publishing replaces, never queues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._latest: Optional[FramePacket] = None

    def publish(self, frame: np.ndarray, self_mask: Optional[np.ndarray]) -> FramePacket:
        with self._lock:
            self._sequence += 1
            packet = FramePacket(self._sequence, time.monotonic(), frame, self_mask)
            self._latest = packet
            return packet

    def latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest


@dataclass
class _Track:
    track_id: int
    label: str
    radius: float
    hits: int = 0
    misses: int = 0
    last_at: float = 0.0
    history: Deque[Tuple[float, float, float, float]] = field(default_factory=deque)
    # history is retained for association, while this short window contains
    # only samples collected after the last clearly non-stationary movement.
    # Using the full history for aiming makes a stopped ball visibly lag.
    stable_history: Deque[Tuple[float, float, float, float]] = field(default_factory=deque)

    @property
    def pos(self) -> Point:
        x, y, _, _ = self.history[-1]
        return (x, y)


class BallTracker:
    """Associate color-labelled balls across frames and expose robust medians.

    A candidate must survive several frames before it can drive geometry.  This
    specifically prevents transient score animations and glyph fragments from
    becoming an aiming target.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def reset_smoothing(self) -> None:
        """Discard moving-history samples while retaining current identities."""
        for track in self._tracks.values():
            if track.history:
                sample = track.history[-1]
                track.history = deque([sample], maxlen=self._history_size())
                track.stable_history = deque([sample], maxlen=self._stable_size())

    def _history_size(self) -> int:
        return max(3, int(getattr(self.cfg, "track_history_frames", 7)))

    def _stable_size(self) -> int:
        """Short stationary window: less jitter without reintroducing lag."""
        configured = int(getattr(self.cfg, "track_stable_window_frames", 3))
        return max(1, min(self._history_size(), configured))

    def update(self, balls: Iterable[Ball], now: Optional[float] = None) -> List[Ball]:
        now = time.monotonic() if now is None else now
        observations = list(balls)
        used: set[int] = set()
        assigned: List[Tuple[Ball, _Track]] = []

        # Match high-confidence observations first.  The gate is intentionally
        # wider than a stationary jitter but far smaller than a table crossing.
        for ball in sorted(observations, key=lambda b: getattr(b, "confidence", 1.0), reverse=True):
            candidates = []
            for track_id, track in self._tracks.items():
                if track_id in used or track.label != ball.label or not track.history:
                    continue
                gate = 2.4 * max(track.radius, ball.radius)
                d = float(np.hypot(ball.pos[0] - track.pos[0], ball.pos[1] - track.pos[1]))
                if d <= gate:
                    candidates.append((d, track_id, track))
            if candidates:
                _, track_id, track = min(candidates, key=lambda item: item[0])
                used.add(track_id)
            else:
                track = _Track(self._next_id, ball.label, float(ball.radius))
                self._tracks[track.track_id] = track
                self._next_id += 1
                used.add(track.track_id)
            previous_sample = track.history[-1] if track.history else None
            previous_at = track.last_at
            track.radius = 0.55 * float(ball.radius) + 0.45 * track.radius
            track.hits += 1
            track.misses = 0
            track.last_at = now
            if track.history.maxlen != self._history_size():
                track.history = deque(track.history, maxlen=self._history_size())
            sample = (float(ball.pos[0]), float(ball.pos[1]), float(ball.radius),
                      float(getattr(ball, "confidence", 1.0)))
            track.history.append(sample)
            if track.stable_history.maxlen != self._stable_size():
                track.stable_history = deque(track.stable_history, maxlen=self._stable_size())

            # The association gate is intentionally wider than a stationary
            # jitter, so a ball can remain on one track while it is still
            # rolling.  Clear the geometry window in that case; only samples
            # after the ball stops are allowed to influence the final point.
            moving = False
            if previous_sample is not None:
                dt = max(1e-3, now - previous_at)
                step = float(np.hypot(sample[0] - previous_sample[0],
                                      sample[1] - previous_sample[1]))
                moving = step / dt > float(getattr(self.cfg, "stationary_speed", 18.0))
            if moving:
                track.stable_history.clear()
            track.stable_history.append(sample)
            assigned.append((ball, track))

        max_misses = max(0, int(getattr(self.cfg, "track_max_misses", 2)))
        for track_id, track in list(self._tracks.items()):
            if track_id not in used:
                track.misses += 1
                if track.misses > max_misses:
                    del self._tracks[track_id]

        confirm = max(1, int(getattr(self.cfg, "track_confirm_frames", 3)))
        stable: List[Ball] = []
        for original, track in assigned:
            if track.hits < confirm:
                continue
            # Do not use the full association history here.  It can contain
            # several frames from before a shot finished, which creates a
            # systematic lag even though TableStateTracker is already READY.
            samples = np.asarray(list(track.stable_history or track.history), dtype=float)
            cx, cy, radius, confidence = np.median(samples, axis=0)
            stable.append(Ball(track.label, (float(cx), float(cy)), float(radius),
                               original.subpixel, float(confidence), track.track_id))

        # Snooker has one ball of every non-red color.  Keep the most established
        # track when a transient duplicate survives candidate generation.
        by_label: Dict[str, List[Ball]] = {}
        for ball in stable:
            by_label.setdefault(ball.label, []).append(ball)
        output: List[Ball] = []
        for label, group in by_label.items():
            if label not in _UNIQUE_LABELS or len(group) == 1:
                output.extend(group)
                continue
            output.append(max(group, key=lambda b: (
                self._tracks[b.track_id].hits, b.confidence
            )))
        return output


class TableState(str, Enum):
    STABILIZING = "stabilizing"
    MOVING = "moving"
    READY = "ready"
    UI_BLOCKED = "ui_blocked"


class TableStateTracker:
    """Hide aim output until ball positions are confirmed stationary."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.state = TableState.STABILIZING
        self._previous: Dict[int, Tuple[Point, float]] = {}
        self._stable_since: Optional[float] = None

    def reset(self) -> None:
        self.state = TableState.STABILIZING
        self._previous.clear()
        self._stable_since = None

    def update(self, balls: Iterable[Ball], now: Optional[float] = None,
               occluded: bool = False) -> TableState:
        now = time.monotonic() if now is None else now
        if occluded:
            self.state = TableState.UI_BLOCKED
            self._stable_since = None
            return self.state
        current = {b.track_id: (b.pos, now) for b in balls if b.track_id >= 0}
        if len(current) < 2:
            self.state = TableState.STABILIZING
            self._stable_since = None
            self._previous = current
            return self.state

        speeds = []
        for track_id, (pos, captured_at) in current.items():
            previous = self._previous.get(track_id)
            if previous is None:
                continue
            prev_pos, prev_at = previous
            dt = max(1e-3, captured_at - prev_at)
            speeds.append(float(np.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1])) / dt)
        self._previous = current
        if not speeds:
            self.state = TableState.STABILIZING
            self._stable_since = None
            return self.state

        speed = max(speeds)
        if speed >= float(getattr(self.cfg, "moving_speed", 85.0)):
            self.state = TableState.MOVING
            self._stable_since = None
            return self.state
        if speed > float(getattr(self.cfg, "stationary_speed", 18.0)):
            self.state = TableState.STABILIZING
            self._stable_since = None
            return self.state

        if self._stable_since is None:
            self._stable_since = now
            self.state = TableState.STABILIZING
            return self.state
        if now - self._stable_since >= float(getattr(self.cfg, "settle_seconds", 0.24)):
            self.state = TableState.READY
        else:
            self.state = TableState.STABILIZING
        return self.state

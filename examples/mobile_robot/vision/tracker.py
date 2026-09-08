"""Small deterministic multi-frame tracker for the showcase detector."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from .types import Detection, VisionResult


@dataclass
class TrackState:
    track_id: str
    label: str
    color: str | None
    first_seen_sim_time: float
    last_seen_sim_time: float
    visible_frame_count: int
    max_confidence: float
    nearest_distance_m: float | None
    last_position_world: tuple[float, ...] | None
    last_bbox_center: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "category": self.label,
            "dominant_color": self.color,
            "first_seen_sim_time": self.first_seen_sim_time,
            "last_seen_sim_time": self.last_seen_sim_time,
            "visible_frame_count": self.visible_frame_count,
            "max_confidence": self.max_confidence,
            "nearest_distance_m": self.nearest_distance_m,
            "last_position_world": list(self.last_position_world) if self.last_position_world else None,
        }


def _bbox_center(detection: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class ObjectTracker:
    """Associate detections by label/color and nearby position."""

    def __init__(self, *, world_match_distance_m: float = 1.25, pixel_match_distance: float = 80.0):
        self.world_match_distance_m = float(world_match_distance_m)
        self.pixel_match_distance = float(pixel_match_distance)
        self._tracks: dict[str, TrackState] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, result: VisionResult) -> VisionResult:
        unmatched = set(self._tracks)
        enriched: list[Detection] = []
        for detection in result.detections:
            track_id = self._find_match(detection, unmatched)
            if track_id is None:
                track_id = f"track-{self._next_id:03d}"
                self._next_id += 1
                self._tracks[track_id] = self._new_state(track_id, detection, result.sim_time)
            else:
                unmatched.discard(track_id)
                self._update_state(self._tracks[track_id], detection, result.sim_time)
            enriched.append(replace(detection, track_id=track_id))

        return replace(result, detections=tuple(enriched))

    def summaries(self) -> list[dict[str, Any]]:
        return [self._tracks[key].to_dict() for key in sorted(self._tracks)]

    def _new_state(self, track_id: str, detection: Detection, sim_time: float) -> TrackState:
        return TrackState(
            track_id=track_id,
            label=detection.label,
            color=detection.color,
            first_seen_sim_time=sim_time,
            last_seen_sim_time=sim_time,
            visible_frame_count=1,
            max_confidence=detection.confidence,
            nearest_distance_m=detection.distance_m,
            last_position_world=detection.position_world,
            last_bbox_center=_bbox_center(detection),
        )

    def _update_state(self, state: TrackState, detection: Detection, sim_time: float) -> None:
        state.last_seen_sim_time = sim_time
        state.visible_frame_count += 1
        state.max_confidence = max(state.max_confidence, detection.confidence)
        if detection.distance_m is not None:
            state.nearest_distance_m = (
                detection.distance_m
                if state.nearest_distance_m is None
                else min(state.nearest_distance_m, detection.distance_m)
            )
        if detection.position_world is not None:
            state.last_position_world = detection.position_world
        state.last_bbox_center = _bbox_center(detection)

    def _find_match(self, detection: Detection, unmatched: set[str]) -> str | None:
        candidates: list[tuple[float, str]] = []
        center = _bbox_center(detection)
        for track_id in unmatched:
            state = self._tracks[track_id]
            if state.label != detection.label or state.color != detection.color:
                continue
            if detection.position_world is not None and state.last_position_world is not None:
                distance = math.dist(detection.position_world[:2], state.last_position_world[:2])
                if distance <= self.world_match_distance_m:
                    candidates.append((distance, track_id))
            else:
                distance = math.dist(center, state.last_bbox_center)
                if distance <= self.pixel_match_distance:
                    candidates.append((distance, track_id))
        return min(candidates)[1] if candidates else None

"""Truth-free visual evidence used to rank open-vocabulary candidates.

The model score is still the source of detection confidence.  This module
only computes weak evidence from the candidate's RGB ROI (colour and coarse
shape) so callers can prefer a better matching candidate or reject a clearly
inconsistent one.  It never reads Genesis masks, object IDs or world poses.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from .open_vocab_grounder import GroundingCandidate


_COLOUR_WORDS = ("yellow", "orange", "red", "green", "blue", "purple", "pink", "black", "white", "gray", "grey")


@dataclass(frozen=True)
class RegionEvidence:
    """Weak prompt-to-ROI compatibility scores in the range ``[0, 1]``."""

    colour_score: float
    shape_score: float

    @property
    def combined_score(self) -> float:
        return 0.65 * self.colour_score + 0.35 * self.shape_score


def _colour_mask(rgb: np.ndarray, colour: str) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    if values.size == 0:
        return np.zeros(values.shape[:2], dtype=bool)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    if colour == "yellow":
        return (red > 0.42) & (green > 0.34) & (blue < 0.34) & (red + green > 2.0 * blue)
    if colour == "orange":
        return (red > 0.42) & (green > 0.16) & (green < 0.95 * red) & (blue < 0.32)
    if colour == "red":
        return (red > 0.35) & (red > 1.25 * green) & (red > 1.25 * blue)
    if colour == "green":
        return (green > 0.18) & (green > 1.08 * red) & (green > 1.08 * blue)
    if colour == "blue":
        return (blue > 0.18) & (blue > 1.08 * red) & (blue > 1.05 * green)
    if colour == "purple":
        return (red > 0.12) & (blue > 0.15) & (blue > 1.05 * green) & (red + blue > 1.5 * green)
    if colour == "pink":
        return (red > 0.35) & (blue > 0.25) & (red > 1.1 * green)
    if colour == "black":
        return values.max(axis=-1) < 0.28
    if colour == "white":
        return values.min(axis=-1) > 0.62
    if colour in {"gray", "grey"}:
        spread = values.max(axis=-1) - values.min(axis=-1)
        return (values.mean(axis=-1) > 0.25) & (spread < 0.12)
    return np.zeros(values.shape[:2], dtype=bool)


def _prompt_colour(prompt: str) -> str | None:
    words = {word.lower() for word in re.findall(r"[a-zA-Z]+", prompt)}
    return next((colour for colour in _COLOUR_WORDS if colour in words), None)


def _shape_score(prompt: str, width: float, height: float) -> float:
    if width <= 0.0 or height <= 0.0:
        return 0.0
    aspect = width / height
    words = set(re.findall(r"[a-zA-Z]+", prompt.lower()))
    if words & {"pillar", "column", "cylinder"}:
        return float(np.clip((1.0 / max(aspect, 1e-6) - 0.8) / 1.4, 0.0, 1.0))
    if words & {"platform", "slab"}:
        return float(np.clip((aspect - 0.5) / 2.5, 0.0, 1.0))
    if words & {"car", "vehicle"}:
        # Cars are usually wider than tall, but perspective and truncation
        # make this deliberately permissive.
        return float(np.clip(1.0 - abs(np.log(max(aspect, 1e-6) / 1.6)) / 1.4, 0.0, 1.0))
    return 0.5


def region_evidence(image: np.ndarray, bbox_xyxy: Iterable[float], prompt: str) -> RegionEvidence:
    """Compute colour/shape evidence for one candidate ROI."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must be an HxWx3 RGB array")
    height, width = array.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    if right <= left or bottom <= top:
        return RegionEvidence(0.0, 0.0)
    # Use the central 80% of the box.  It reduces wall/floor contamination
    # without requiring a segmentation model.
    margin_x = max(0, int((right - left) * 0.1))
    margin_y = max(0, int((bottom - top) * 0.1))
    roi = array[top + margin_y : bottom - margin_y or bottom, left + margin_x : right - margin_x or right]
    colour = _prompt_colour(prompt)
    if colour is None:
        colour_score = 0.5
    else:
        colour_score = float(_colour_mask(roi, colour).mean()) if roi.size else 0.0
    shape_score = _shape_score(prompt, right - left, bottom - top)
    return RegionEvidence(colour_score=colour_score, shape_score=shape_score)


def candidate_rank_score(image: np.ndarray, candidate: GroundingCandidate) -> float:
    """Return a conservative score for ordering candidates within one prompt."""

    evidence = region_evidence(image, candidate.bbox_xyxy, candidate.prompt)
    # Never turn weak ROI evidence into a detection.  It only changes ordering
    # among candidates already accepted by the model confidence threshold.
    return float(candidate.confidence * (0.7 + 0.3 * evidence.combined_score))


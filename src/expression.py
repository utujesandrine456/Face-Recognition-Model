# src/expression.py
"""
Lightweight expression cues from MediaPipe FaceMesh landmarks:
  - smiling / not smiling
  - eyes open / eyes closed
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


# 6-point eye contours (better closed-eye sensitivity than 2 points)
# p1 outer, p2/p3 upper, p4 inner, p5/p6 lower
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (263, 387, 385, 362, 380, 373)

MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOT = 13, 14
L_EYE_OUTER, R_EYE_OUTER = 33, 263

# Running open-eye baseline (adaptive closed threshold)
_open_ear_ema: Optional[float] = None
_eyes_closed_latched: bool = False

# Blink tracking: open -> closed -> open within a short window
_blink_was_closed: bool = False
_blink_closed_t: float = 0.0
_blink_count: int = 0


@dataclass
class ExpressionState:
    smiling: bool
    eyes_closed: bool
    eyes_open: bool
    eye_openness: float  # EAR
    smile_score: float
    label: str
    blinked: bool = False  # True on the frame a blink completes
    blink_count: int = 0

    def as_log(self) -> str:
        eye = "eyes CLOSED" if self.eyes_closed else "eyes OPEN"
        smile = "SMILING" if self.smiling else "not smiling"
        parts = [smile, eye]
        if self.blinked:
            parts.append("BLINK")
        return ", ".join(parts)


@dataclass
class FaceSideState:
    side: str  # "LEFT" | "CENTER" | "RIGHT"
    offset: float  # -0.5 .. +0.5 (negative = left of frame center)
    label: str

    def as_log(self) -> str:
        return f"on the {self.side}"


def face_horizontal_side(
    face_cx: float,
    frame_width: int,
    center_frac: float = 0.12,
) -> FaceSideState:
    """
    Classify face center X as LEFT / CENTER / RIGHT in the frame.
    center_frac is the deadzone around mid-frame treated as CENTER.
    """
    if frame_width <= 1:
        return FaceSideState(side="CENTER", offset=0.0, label="CENTER")
    t = float(face_cx) / float(frame_width)
    t = max(0.0, min(1.0, t))
    offset = t - 0.5
    if abs(offset) <= float(center_frac):
        side = "CENTER"
    elif offset < 0:
        side = "LEFT"
    else:
        side = "RIGHT"
    return FaceSideState(side=side, offset=float(offset), label=side)


def _xy(lm: Sequence[Any], idx: int, w: float, h: float) -> np.ndarray:
    p = lm[idx]
    return np.array([float(p.x) * w, float(p.y) * h], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_ear(lm: Sequence[Any], idxs: Sequence[int], w: float, h: float) -> float:
    p1, p2, p3, p4, p5, p6 = [_xy(lm, i, w, h) for i in idxs]
    vert = _dist(p2, p6) + _dist(p3, p5)
    horiz = 2.0 * _dist(p1, p4) + 1e-6
    return vert / horiz


def reset_eye_baseline() -> None:
    global _open_ear_ema, _eyes_closed_latched
    global _blink_was_closed, _blink_closed_t, _blink_count
    _open_ear_ema = None
    _eyes_closed_latched = False
    _blink_was_closed = False
    _blink_closed_t = 0.0
    _blink_count = 0


def _update_blink(eyes_closed: bool, now: float, min_s: float = 0.06, max_s: float = 0.55) -> bool:
    """
    Detect a blink as eyes going closed then open again within [min_s, max_s].
    Returns True on the reopen frame when a blink is counted.
    """
    global _blink_was_closed, _blink_closed_t, _blink_count

    blinked = False
    if eyes_closed:
        if not _blink_was_closed:
            _blink_closed_t = now
        _blink_was_closed = True
    else:
        if _blink_was_closed:
            closed_for = now - _blink_closed_t
            if min_s <= closed_for <= max_s:
                _blink_count += 1
                blinked = True
        _blink_was_closed = False
    return blinked


def analyze_facemesh_expression(
    landmarks: Sequence[Any],
    frame_w: int,
    frame_h: int,
    smile_thresh: float = 0.42,
    now: Optional[float] = None,
) -> ExpressionState:
    """
    landmarks: mediapipe face_mesh landmark list (normalized x,y).
    """
    global _open_ear_ema, _eyes_closed_latched

    now_t = float(now) if now is not None else time.time()
    w, h = float(frame_w), float(frame_h)

    left_ear = _eye_ear(landmarks, LEFT_EYE, w, h)
    right_ear = _eye_ear(landmarks, RIGHT_EYE, w, h)
    ear = 0.5 * (left_ear + right_ear)

    # Adaptive baseline from open-looking frames (FaceMesh rarely goes near 0 when closed).
    if _open_ear_ema is None:
        _open_ear_ema = max(ear, 0.28)
    if ear >= 0.26:
        _open_ear_ema = 0.85 * _open_ear_ema + 0.15 * ear

    # Closed if clearly below the person's open baseline (with hysteresis).
    close_thresh = max(0.18, min(0.32, 0.78 * float(_open_ear_ema)))
    open_thresh = close_thresh + 0.04

    if _eyes_closed_latched:
        eyes_closed = ear < open_thresh
    else:
        eyes_closed = ear < close_thresh
    _eyes_closed_latched = bool(eyes_closed)
    eyes_open = not eyes_closed

    blinked = _update_blink(eyes_closed, now_t)

    # Smile: wide mouth relative to eye span + corners raised vs lip center
    ml = _xy(landmarks, MOUTH_LEFT, w, h)
    mr = _xy(landmarks, MOUTH_RIGHT, w, h)
    mt = _xy(landmarks, MOUTH_TOP, w, h)
    mb = _xy(landmarks, MOUTH_BOT, w, h)
    le = _xy(landmarks, L_EYE_OUTER, w, h)
    re = _xy(landmarks, R_EYE_OUTER, w, h)

    mouth_w = _dist(ml, mr)
    eye_span = _dist(le, re) + 1e-6
    mouth_h = _dist(mt, mb)
    width_ratio = mouth_w / eye_span
    mouth_cy = 0.5 * (mt[1] + mb[1])
    corner_lift = (mouth_cy - 0.5 * (ml[1] + mr[1])) / (eye_span + 1e-6)
    smile_score = float(0.65 * width_ratio + 0.35 * max(0.0, corner_lift * 8.0))
    if mouth_h / (mouth_w + 1e-6) > 0.55:
        smile_score *= 0.7
    smiling = smile_score >= smile_thresh

    parts = [
        "SMILING" if smiling else "not smiling",
        "eyes CLOSED" if eyes_closed else "eyes OPEN",
    ]
    if blinked:
        parts.append("BLINK")
    label = " | ".join(parts)

    return ExpressionState(
        smiling=bool(smiling),
        eyes_closed=bool(eyes_closed),
        eyes_open=bool(eyes_open),
        eye_openness=float(ear),
        smile_score=float(smile_score),
        label=label,
        blinked=bool(blinked),
        blink_count=int(_blink_count),
    )


def analyze_optional(
    landmarks: Optional[Sequence[Any]],
    frame_w: int,
    frame_h: int,
) -> Optional[ExpressionState]:
    if landmarks is None:
        return None
    try:
        return analyze_facemesh_expression(landmarks, frame_w, frame_h)
    except Exception:
        return None

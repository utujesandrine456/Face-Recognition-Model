# src/recognize.py
"""
Multi-face recognition (CPU-friendly):
Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.

Run: python -m src.recognize
     python -m src.recognize --cam 1   # use USB / embedded camera (not laptop)

Keys:
  q : quit
  r : reload DB from disk (data/db/face_db.npz)
  +/- : adjust threshold (distance) live
  d : toggle debug overlay
  i : invert pan mapping (if servo turns the wrong way)
  m : toggle MQTT pan publish on/off
  s : toggle search-when-lost on/off
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

from .haar_5pt import align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .mqtt_servo import ServoPanPublisher, PanController
from .expression import ExpressionState, FaceSideState, analyze_optional, face_horizontal_side


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords
    expression: Optional[ExpressionState] = None


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray, pad_x: float = 0.55, pad_y_top: float = 0.85, pad_y_bot: float = 1.15,
) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    # Softer eye-span check so closed-eye frames are not rejected as often.
    if eye_dist < float(min_eye_dist) * 0.75:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
    return out


class HaarFaceMesh5pt:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.75,
        debug: bool = False,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = float(smooth_alpha)
        self._prev_kps: Optional[np.ndarray] = None
        self._prev_box: Optional[np.ndarray] = None

        if haar_xml is None:
            haar_xml = "models/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe==0.10.21"
            )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def _facemesh_5pt_full(self, frame_bgr: np.ndarray):
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None, None

        lm = res.multi_face_landmarks[0].landmark
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP, self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]
        pts = []
        for i in idxs:
            p = lm[i]
            pts.append([p.x * W, p.y * H])
        kps = np.array(pts, dtype=np.float32)

        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        expr = analyze_optional(lm, W, H)
        return kps, expr

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            self._prev_kps = None
            self._prev_box = None
            return []

        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]

        # Full-frame FaceMesh is much more stable than per-ROI mesh on USB cams.
        kps, expr = self._facemesh_5pt_full(frame_bgr)
        if kps is None:
            return []

        x, y, w, h = faces[0].tolist()
        margin = 0.45
        x1m, y1m = x - margin * w, y - margin * h
        x2m, y2m = x + (1.0 + margin) * w, y + (1.0 + margin) * h
        inside = (
            (kps[:, 0] >= x1m) & (kps[:, 0] <= x2m) &
            (kps[:, 1] >= y1m) & (kps[:, 1] <= y2m)
        )
        if float(inside.mean()) < 0.50:
            if self.debug:
                print("[recognize] FaceMesh not consistent with Haar -> skip")
            return []

        if not _kps_span_ok(kps, min_eye_dist=max(8.0, 0.12 * float(w))):
            if self.debug:
                print("[recognize] 5pt geometry failed -> skip")
            return []

        bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
        kps_s = _ema(self._prev_kps, kps, self.smooth_alpha)
        bb_s = _ema(self._prev_box, bb, self.smooth_alpha)
        self._prev_kps = kps_s.copy()
        self._prev_box = bb_s.copy()

        x1, y1, x2, y2 = _clip_xyxy(bb_s[0], bb_s[1], bb_s[2], bb_s[3], W, H)
        return [
            FaceDet(
                x1=x1, y1=y1, x2=x2, y2=y2, score=1.0,
                kps=kps_s.astype(np.float32), expression=expr,
            )
        ]


class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.45):
        self.db = db
        self.dist_thresh = float(dist_thresh)
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())
        if self._names:
            self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)

        e = emb.reshape(1, -1).astype(np.float32)
        sims = (self._mat @ e.T).reshape(-1)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim
        ok = best_dist <= self.dist_thresh

        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=float(best_dist),
            similarity=float(best_sim),
            accepted=bool(ok),
        )


def main():
    parser = argparse.ArgumentParser(description="Live face recognition + MQTT servo pan")
    parser.add_argument(
        "--cam",
        type=int,
        default=1,
        help="OpenCV camera index (default 1 = usually USB/embedded; 0 = laptop webcam)",
    )
    args = parser.parse_args()

    db_path = Path("data/db/face_db.npz")

    det = HaarFaceMesh5pt(min_size=(60, 60), smooth_alpha=0.75, debug=False)
    embedder = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False)

    db = load_db_npz(db_path)
    # Slightly looser threshold helps smile / blink variation after diverse enrollment.
    matcher = FaceDBMatcher(db=db, dist_thresh=0.45)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera index {args.cam} not available. "
            "Find the right one with: .\\.venv\\Scripts\\python.exe -m src.camera --list"
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mqtt_enabled = True
    search_enabled = True
    invert_pan = False
    pan_angle: Optional[int] = 90
    pan_mode = "idle"
    pan = PanController(
        deadzone_frac=0.07,
        track_gain=22.0,
        max_step_deg=3.0,
        lost_before_search_s=0.45,
        search_step_deg=5.0,
        search_period_s=0.10,
    )
    servo_pub: Optional[ServoPanPublisher] = None
    locked_label: Optional[str] = None
    # Hold last accepted identity briefly through blinks / expression changes.
    sticky_name: Optional[str] = None
    sticky_until = 0.0
    sticky_hold_s = 0.7
    last_expr_log: Optional[str] = None
    last_side_log: Optional[str] = None
    try:
        servo_pub = ServoPanPublisher(min_publish_interval_s=0.08, min_angle_delta=1)
        servo_pub.connect()
        servo_pub.publish_angle(90, force=True)
    except Exception as e:
        mqtt_enabled = False
        print(f"[recognize] MQTT disabled: {e}")

    print(
        f"Recognize cam={args.cam}. "
        "q=quit, r=reload, +/- thr, d=debug, m=MQTT, i=invert, s=search"
    )
    print("[pan] face -> LOCK. missing ~0.45s -> SEARCH sweep. Watch mode= on screen.")

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=5)
        vis = frame.copy()

        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        h, w = vis.shape[:2]
        thumb = 112
        pad = 8
        x0 = w - thumb - pad
        y0 = 80
        shown = 0

        # Prefer a known face for pan locking; else use the first detection.
        track_face: Optional[FaceDet] = None
        track_label = "none"
        first_face: Optional[FaceDet] = None
        first_label = "none"
        known_face: Optional[FaceDet] = None
        known_label = "none"

        for i, f in enumerate(faces):
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = embedder.embed(aligned).embedding
            mr = matcher.match(emb)

            now_t = time.time()
            if mr.accepted and mr.name is not None:
                sticky_name = mr.name
                sticky_until = now_t + sticky_hold_s
                label = mr.name
                accepted = True
            elif sticky_name is not None and now_t < sticky_until:
                # Keep identity through blink / short expression change.
                label = sticky_name
                accepted = True
            else:
                label = mr.name if mr.name is not None else "Unknown"
                accepted = bool(mr.accepted)

            line1 = f"{label}"
            line2 = f"dist={mr.distance:.3f} sim={mr.similarity:.3f}"

            color = (0, 255, 0) if accepted else (0, 0, 255)
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 2, color, -1)
            cv2.putText(vis, line1, (f.x1, max(0, f.y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(vis, line2, (f.x1, max(0, f.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if f.expression is not None:
                face_mid_x = 0.5 * (f.x1 + f.x2)
                side = face_horizontal_side(face_mid_x, w, center_frac=0.12)
                expr_txt = f"{f.expression.label} | {side.label}"
                cv2.putText(
                    vis, expr_txt, (f.x1, min(h - 10, f.y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                )
                log_key = f"{label}|{expr_txt}"
                if log_key != last_expr_log:
                    last_expr_log = log_key
                    print(
                        f"[expression] {label}: {f.expression.as_log()}, {side.as_log()} "
                        f"(smile={f.expression.smile_score:.2f}, ear={f.expression.eye_openness:.2f}, "
                        f"offset={side.offset:+.2f})"
                    )
                if f.expression.blinked:
                    print(f"[blink] {label}: BLINK detected (count={f.expression.blink_count})")
                    cv2.putText(
                        vis, "BLINK!", (f.x1, min(h - 10, f.y2 + 44)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2,
                    )
                if side.side != last_side_log:
                    last_side_log = side.side
                    print(f"[position] {label}: {side.as_log()}")
            else:
                face_mid_x = 0.5 * (f.x1 + f.x2)
                side = face_horizontal_side(face_mid_x, w, center_frac=0.12)
                cv2.putText(
                    vis, f"side: {side.label}", (f.x1, min(h - 10, f.y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                )
                if side.side != last_side_log:
                    last_side_log = side.side
                    print(f"[position] {label}: {side.as_log()}")

            if y0 + thumb <= h and shown < 4:
                vis[y0:y0 + thumb, x0:x0 + thumb] = aligned
                cv2.putText(vis, f"{i+1}:{label}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y0 += thumb + pad
                shown += 1

            if show_debug:
                dbg = f"kpsLeye=({f.kps[0,0]:.0f},{f.kps[0,1]:.0f})"
                cv2.putText(vis, dbg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if first_face is None:
                first_face = f
                first_label = label
            if accepted and known_face is None:
                known_face = f
                known_label = label
            # Stick to previously locked identity when possible.
            if locked_label is not None and label == locked_label:
                track_face = f
                track_label = label

        if track_face is None:
            if known_face is not None:
                track_face = known_face
                track_label = known_label
            elif first_face is not None:
                track_face = first_face
                track_label = first_label

        face_cx: Optional[float] = None
        if track_face is not None:
            face_cx = 0.5 * (track_face.x1 + track_face.x2)
            cv2.line(vis, (int(face_cx), 0), (int(face_cx), h), (255, 200, 0), 2)
            cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
            dz = int(0.04 * w)
            cv2.line(vis, (w // 2 - dz, 0), (w // 2 - dz, h), (80, 80, 80), 1)
            cv2.line(vis, (w // 2 + dz, 0), (w // 2 + dz, h), (80, 80, 80), 1)

        pan.set_invert(invert_pan)
        prev_angle = int(pan_angle) if pan_angle is not None else None
        pan_angle, pan_mode = pan.update(face_cx, w, allow_search=search_enabled)

        if pan_mode == "lock" and track_label != "none":
            locked_label = track_label
        elif pan_mode in ("search", "idle"):
            locked_label = None

        if mqtt_enabled and servo_pub is not None:
            if pan_mode == "search" and pan_angle != prev_angle:
                # Force each new search step (bypass rate limits).
                servo_pub.publish_angle(int(pan_angle), force=True)
            elif pan_mode == "lock" and pan_angle != prev_angle:
                servo_pub.publish_angle(int(pan_angle), force=False)

        # Big lock / search status
        if pan_mode == "lock":
            status = f"LOCKED: {track_label}"
            status_color = (0, 255, 0)
        elif pan_mode == "search":
            status = "SEARCHING..."
            status_color = (0, 165, 255)
        elif pan_mode == "hold":
            status = "HOLD (lost)"
            status_color = (0, 255, 255)
        else:
            status = "IDLE"
            status_color = (200, 200, 200)
        cv2.putText(vis, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        header = f"cam={args.cam} IDs={len(matcher._names)} thr(dist)={matcher.dist_thresh:.2f}"
        if fps is not None:
            header += f" fps={fps:.1f}"
        header += f" pan={pan_angle} mode={pan_mode}"
        header += f" mqtt={'ON' if mqtt_enabled else 'OFF'}"
        header += f" search={'ON' if search_enabled else 'OFF'}"
        if invert_pan:
            header += " inv"
        cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("recognize_new", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            matcher.reload_from(db_path)
            print(f"[recognize] reloaded DB: {len(matcher._names)} identities")
        elif key in (ord("+"), ord("=")):
            matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
            print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
        elif key == ord("-"):
            matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
            print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")
        elif key == ord("m"):
            mqtt_enabled = not mqtt_enabled
            print(f"[recognize] MQTT pan: {'ON' if mqtt_enabled else 'OFF'}")
        elif key == ord("i"):
            invert_pan = not invert_pan
            pan.set_invert(invert_pan)
            print(f"[recognize] pan invert: {'ON' if invert_pan else 'OFF'}")
        elif key == ord("s"):
            search_enabled = not search_enabled
            print(f"[recognize] search-when-lost: {'ON' if search_enabled else 'OFF'}")

    if servo_pub is not None:
        servo_pub.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

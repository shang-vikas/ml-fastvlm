# pugsy_ai/pipelines/fastvlm/video_utils.py

import cv2
import numpy as np
import logging
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger("fastvlm.video_utils")


# ------------------------------
# Scene fallback helper
# ------------------------------
def ensure_scenes_fallback(
    scenes: List[Tuple[float, float]],
    stats: Dict[str, Any],
    strategy: str = "single_scene"
) -> List[Tuple[float, float]]:
    """
    Fallback strategies:
        - "single_scene": return one big scene (0 -> duration)
        - "none": return scenes unchanged
    """
    if scenes:
        return scenes

    if strategy == "none":
        return scenes

    duration = float(stats.get("duration_sec", 0.0) or 0.0)
    logger.warning("No scenes detected — falling back to full video scene")
    return [(0.0, max(0.0, duration))]


# ------------------------------
# Uniform sampling helper
# ------------------------------
def sample_uniform_frames(
    video_path: str,
    target_frame_count: int,
    max_res: Optional[int],
    stats: Dict[str, Any]
) -> List[Tuple[float, Any]]:
    """
    Return up to `target_frame_count` frames sampled uniformly.
    Each frame is returned as (timestamp_sec, bgr_numpy_frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames_total / fps if frames_total > 0 else stats.get("duration_sec", 0.0) or 0.0
        duration = float(duration)

        target = max(1, int(target_frame_count))
        timestamps = np.linspace(0, duration, target)

        samples: List[Tuple[float, Any]] = []
        for ts in timestamps:
            frame_idx = int(ts * fps)
            frame_idx = min(max(frame_idx, 0), max(frames_total - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Resize if needed
            if max_res:
                h, w = frame.shape[:2]
                long_side = max(h, w)
                if long_side > max_res:
                    scale = max_res / long_side
                    frame = cv2.resize(
                        frame,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
            samples.append((ts, frame))

        return samples
    finally:
        cap.release()

def compute_fallback_frame_count(
    duration_sec: float,
    min_frames: int,
    max_frames: int,
    seconds_per_frame: float
) -> int:
    """
    Returns how many frames to sample when normal extract returns none.

    Logic:
      - 1 frame per `seconds_per_frame`
      - But always at least `min_frames`
      - And never more than `max_frames`
    """
    if duration_sec <= 0:
        return min_frames

    est = int(duration_sec / seconds_per_frame)
    est = max(est, min_frames)
    est = min(est, max_frames)
    return est

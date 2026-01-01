# fastvlm_config.py
import os
import logging
from dataclasses import dataclass
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


logger = logging.getLogger("FastVLMConfig")


@dataclass
class FastVLMConfig:
    model_path: str
    device: str = "cuda"
    scene_threshold: float = 30.0
    frame_similarity_threshold: float = 0.90
    max_video_seconds: float = 90.0
    max_resolution: int = 1080
    max_frames: int = 24
    max_context_chars: int = 256
    enable_summary: bool = True
    enable_analysis: bool = False
    log_level: str = "INFO"

    # collage / sampling controls
    max_collages: int = 4              # hard cap on VLM calls
    frames_per_collage: int = 4        # fixed for 2x2
    scene_sample_rate: int = 3         # frames sampled per scene
    motion_threshold: float = 0.25     # cosine delta threshold
    min_scene_duration_sec: float = 0.5



def _bool_from_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def load_fastvlm_config(toml_path: Optional[str] = None) -> FastVLMConfig:
    """
    Load config from TOML file, then apply env overrides.
    """
    toml_path = toml_path or os.getenv("FASTVLM_CONFIG_PATH", "fastvlm.toml")

    data = {}
    if os.path.exists(toml_path):
        with open(toml_path, "rb") as f:
            raw = tomllib.load(f)
        data = raw.get("fastvlm", {})
    else:
        logger.warning("Config TOML not found at %s, using defaults + env overrides", toml_path)

    def g(key, default):
        return data.get(key, default)

    cfg = FastVLMConfig(
        model_path=os.getenv("FASTVLM_MODEL_PATH", g("model_path", "")),
        device=os.getenv("FASTVLM_DEVICE", g("device", "cuda")),

        scene_threshold=float(
            os.getenv("FASTVLM_SCENE_THRESHOLD", g("scene_threshold", 30.0))
        ),

        frame_similarity_threshold=float(
            os.getenv("FASTVLM_FRAME_SIM_THRESHOLD", g("frame_similarity_threshold", 0.90))
        ),

        max_video_seconds=float(
            os.getenv("FASTVLM_MAX_VIDEO_SEC", g("max_video_seconds", 90.0))
        ),

        max_resolution=int(
            os.getenv("FASTVLM_MAX_RES", g("max_resolution", 1080))
        ),

        max_frames=int(
            os.getenv("FASTVLM_MAX_FRAMES", g("max_frames", 24))
        ),

        max_context_chars=int(
            os.getenv("FASTVLM_MAX_CONTEXT_CHARS", g("max_context_chars", 256))
        ),

        enable_summary=_bool_from_env(
            "FASTVLM_ENABLE_SUMMARY",
            g("enable_summary", True),
        ),

        enable_analysis=_bool_from_env(
            "FASTVLM_ENABLE_ANALYSIS",
            g("enable_analysis", False),
        ),

        log_level=os.getenv(
            "FASTVLM_LOG_LEVEL",
            g("log_level", "INFO"),
        ),

        # -----------------------------
        # Collage / sampling controls
        # -----------------------------

        # Hard cap on FastVLM calls per video
        max_collages=int(
            os.getenv("FASTVLM_MAX_COLLAGES", g("max_collages", 5))
        ),

        # Frames per collage (2x2 grid = 4)
        frames_per_collage=int(
            os.getenv("FASTVLM_FRAMES_PER_COLLAGE", g("frames_per_collage", 4))
        ),

        # Frames sampled per scene *before* motion filtering
        scene_sample_rate=int(
            os.getenv("FASTVLM_SCENE_SAMPLE_RATE", g("scene_sample_rate", 3))
        ),

        # Motion threshold for frame-to-frame inclusion
        motion_threshold=float(
            os.getenv("FASTVLM_MOTION_THRESHOLD", g("motion_threshold", 0.15))
        ),

        # Ignore very short scenes (noise cuts)
        min_scene_duration_sec=float(
            os.getenv("FASTVLM_MIN_SCENE_DURATION_SEC", g("min_scene_duration_sec", 1.0))
        ),
    )

    if not cfg.model_path:
        raise RuntimeError(
            "FASTVLM model_path is not configured (TOML or FASTVLM_MODEL_PATH missing)."
        )

    logger.info(f"cfg - {cfg}")
    return  cfg


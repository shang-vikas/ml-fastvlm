# core_fastvlm_engine.py
"""
FastVLM stage-0 engine (text-first per-frame captions).

This file is intentionally minimal for Stage-0:
 - extract key frames
 - call FastVLM once per frame with a short human-readable prompt
 - return per-frame captions (1-2 short sentences)

It includes small, well-documented hooks for Stage-1 (category/classifier)
and Stage-2 (category-specific tagger) so those can be added later.
"""
import json
import gc
import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

from .fastvlm_config import FastVLMConfig, load_fastvlm_config

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Prompts: we use a human-readable single-frame prompt for Stage-0
from .prompts import HUMAN_FRAME_PROMPT

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=os.getenv("FASTVLM_LOG_LEVEL", "INFO"),
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FastVLMEngine")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# -----------------------
# Quantization helpers (kept for runtime)
# -----------------------
SKIP_MODULES = ["vision", "clip", "projector", "lm_head", "embed", "norm"]


def quantize_model_8bit(model: torch.nn.Module) -> torch.nn.Module:
    """
    Convert linear layers to 8-bit (except vision & projector etc.).
    Safe to call; will fallback if bitsandbytes not available at runtime.
    """
    try:
        import bitsandbytes as bnb
        import torch.nn as nn
    except Exception:
        logger.warning("bitsandbytes not available — skipping 8-bit quantization.")
        return model

    for name, module in model.named_children():
        if any(skip in name.lower() for skip in SKIP_MODULES):
            continue

        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            device = module.weight.device

            logger.info(f"Quantizing layer {name} to 8-bit on {device} ...")
            qlayer = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=bias,
                device=device,
            )
            qlayer.weight.data = module.weight.data
            if bias and module.bias is not None:
                qlayer.bias.data = module.bias.data
            setattr(model, name, qlayer)
        else:
            if len(list(module.children())) > 0:
                quantize_model_8bit(module)

    return model


def check_layers(model: torch.nn.Module) -> None:
    """
    Basic sanity check, logs some info about layer devices & dtypes.
    """
    total_params = 0
    cuda_params = 0
    for n, p in model.named_parameters():
        total_params += p.numel()
        if p.is_cuda:
            cuda_params += p.numel()
    logger.info("Model parameters: total=%d, cuda=%d", total_params, cuda_params)

# -----------------------
# Summarizer / Analyzer / Tag heuristics (lightweight)
# -----------------------
_summarizer = None
_analyzer = None

COLLAGE_SUMMARY_PROMPT = (
"""You are given an image containing four frames from the same video,
arranged in chronological order from earliest to latest.

Summarize what happens in the video based on these frames.
Focus on actions and activities.
Do not describe the image layout or the frames individually.
Write one concise summary sentence."""
)

COLLAGE_SUMMARY_PROMPT_v1 = (
    """You are given an image containing multiple moments from a video,
ordered from earlier to later.

Describe what is happening in the video.
Focus on visible actions and changes only.
Do not infer intent, emotion, or meaning.
Write one concise sentence."""
)


COLLAGE_SUMMARY_PROMPT_v2 = ("""You are given an image containing multiple frames from the same video,
arranged from earlier to later.

Task:
1. Describe the visible actions or activities.
2. List any clearly visible on-screen text exactly as written.
3. Do NOT infer motivation, meaning, or transformation.
4. Do NOT summarize or explain the message.

Return your answer in JSON with keys:
- "visible_text": list of strings
- "actions": list of short phrases
                             
Rules:
- Do NOT infer intent, motivation, or meaning.
- Do NOT merge text into sentences.
- Do NOT explain the image.
- Output valid JSON only. No extra text.
""")


COLLAGE_SUMMARY_PROMPT_v3 = """
You are given an image containing multiple moments from a video,
ordered from earlier to later.

Extract only what is directly visible.

Return a JSON object with exactly these fields:
- "visible_text": list of all readable on-screen text (captions, overlays, titles)
- "actions": list of physical actions that are visually observable
  (e.g. walking, lifting, speaking into a microphone)

Rules:
- If no physical actions are visible, return an empty list for "actions"
- Do NOT convert text into actions
- Do NOT infer intent, transformation, or meaning
- Do NOT describe layout or camera angles
- Output valid JSON only
"""



def get_summarizer(device: str):
    global _summarizer
    if _summarizer is not None:
        return _summarizer

    try:
        from transformers import pipeline
    except Exception:
        logger.warning("transformers not installed; summarize_captions will use a simple join fallback.")
        return None

    logger.info("Loading summarization model (flan-t5-base -> pegasus-xsum fallback) ...")
    try:
        _summarizer = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        )
    except Exception:
        logger.exception("Failed to load flan-t5-base, falling back to pegasus-xsum.")
        try:
            _summarizer = pipeline(
                "summarization",
                model="google/pegasus-xsum",
                device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            )
        except Exception:
            logger.exception("Failed to load summarization models; using fallback join.")
            _summarizer = None
    return _summarizer

def dedup_events(captions: list[str]) -> list[str]:
    events = []
    last = None

    for c in captions:
        c = c.strip()
        if not c:
            continue

        # normalize lightly
        norm = c.lower()
        norm = norm.replace("a person", "").replace("the person", "")
        norm = norm.replace("someone", "").strip()

        if norm == last:
            continue

        events.append(c)
        last = norm

    return events


# def summarize_captions(cfg: FastVLMConfig, captions: List[str]) -> str:
#     """
#     Turn per-frame captions (list of strings) into a short video-level summary.
#     If cfg.enable_summary is False or no summarizer available, returns a safe join/truncate.
#     """
#     if not cfg.enable_summary or not captions:
#         return " ".join(captions).strip()

#     summarizer = get_summarizer(cfg.device)
#     if summarizer is None:
#         # fallback: naive join + truncate
#         joined = " ".join(captions)
#         return joined[: cfg.max_context_chars].strip()

#     joined = " ".join(captions)
#     if len(joined) > cfg.max_context_chars:
#         joined = joined[: cfg.max_context_chars]

#     prompt = f"Summarize what is happening in this short social media video:\n{joined}"
#     try:
#         out = summarizer(prompt, max_new_tokens=64, num_return_sequences=1)[0]
#         summary = out.get("generated_text") or out.get("summary_text") or ""
#         return summary.strip()
#     except Exception:
#         logger.exception("Summarizer pipeline failed; returning truncated join.")
#         return joined[: cfg.max_context_chars].strip()

def summarize_captions(cfg: FastVLMConfig, captions: List[str]) -> str:
    if not captions:
        return ""

    # 1. Deduplicate into events
    events = dedup_events(captions)

    if not events:
        return ""

    # 2. Keep last N events (late events matter more)
    # MAX_EVENTS = 10
    # events = events[-MAX_EVENTS:]

    # 3. Build temporal, action-biased input
    timeline = "\n".join(
        f"[event {i+1}] {evt}"
        for i, evt in enumerate(events)
    )

    prompt = (
        "The following are observations from different moments in a video.\n"
        "Each line represents a separate event and may be incomplete.\n"
        "Infer what happens over time.\n"
        "Focus on actions and outcomes, not static appearance.\n"
        "Write a concise summary of the video.\n\n"
        f"{timeline}"
    )

    summarizer = get_summarizer(cfg.device)
    if summarizer is None:
        return " ".join(events)

    try:
        out = summarizer(
            prompt,
            max_new_tokens=64,
            num_return_sequences=1,
        )[0]
        return (
            out.get("generated_text")
            or out.get("summary_text")
            or ""
        ).strip()
    except Exception:
        return " ".join(events)



def get_analyzer(device: str):
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    try:
        from transformers import pipeline
    except Exception:
        logger.warning("transformers not installed; analyze_summary will be disabled.")
        return None

    logger.info("Loading analyzer model (microsoft/phi-3-mini-4k-instruct) ...")
    try:
        _analyzer = pipeline(
            "text-generation",
            model="microsoft/phi-3-mini-4k-instruct",
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        )
    except Exception:
        logger.exception("Failed to load analyzer model; analyze_summary will be disabled.")
        _analyzer = None
    return _analyzer


def analyze_summary(cfg: FastVLMConfig, summary: str) -> Optional[Dict[str, Any]]:
    """
    Optional: ask an analyzer model to extract structured attributes from the summary.
    Returns a dict or None on failure / disabled.
    """
    if not cfg.enable_analysis or not summary:
        return None

    analyzer = get_analyzer(cfg.device)
    if analyzer is None:
        return None

    system_prompt = (
        "You are an assistant that analyzes short social media videos and returns a concise JSON "
        "with keys: 'video_type', 'setting', 'activity', 'people_count', 'vibe'. "
        "No explanation, only JSON."
    )
    user_prompt = f"Video description: {summary}"
    prompt = system_prompt + "\n" + user_prompt
    try:
        out = analyzer(prompt, max_new_tokens=128, do_sample=False)[0].get("generated_text", "")
        start = out.index("{")
        end = out.rindex("}") + 1
        return json.loads(out[start:end])
    except Exception:
        logger.exception("Failed to parse analyzer output as JSON: %s", out if 'out' in locals() else '')
        return None


def tag_from_text(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Lightweight heuristic tagger used for MVP.
    Returns (visual_tags, vibe_tags, safety_flags).
    """
    text_l = (text or "").lower()
    visual_tags: List[str] = []
    vibe_tags: List[str] = []
    safety_flags: List[str] = []

    if any(w in text_l for w in ["gym", "workout", "exercise", "lift", "squat"]):
        visual_tags.append("gym")
    if any(w in text_l for w in ["beach", "ocean", "sea"]):
        visual_tags.append("beach")
    if any(w in text_l for w in ["office", "meeting", "laptop"]):
        visual_tags.append("office")
    if any(w in text_l for w in ["travel", "trip", "flight", "airport"]):
        visual_tags.append("travel")
    if any(w in text_l for w in ["food", "recipe", "cook", "cooking"]):
        visual_tags.append("food")
    if any(w in text_l for w in ["dog", "cat", "pet"]):
        visual_tags.append("pets")

    if any(w in text_l for w in ["tutorial", "how to", "guide", "tips"]):
        vibe_tags.append("educational")
    if any(w in text_l for w in ["motivational", "inspiring", "inspiration"]):
        vibe_tags.append("motivational")
    if any(w in text_l for w in ["funny", "lol", "joke", "meme"]):
        vibe_tags.append("funny")

    if any(w in text_l for w in ["blood", "gun", "fight", "kill"]):
        safety_flags.append("violence")
    if any(w in text_l for w in ["nude", "sexual", "nsfw"]):
        safety_flags.append("sexual")

    return visual_tags, vibe_tags, safety_flags


# -----------------------
# FastVLM core model wrapper
# -----------------------
class FastVLMModel:
    """
    Thin wrapper around LLaVA/FastVLM that exposes a single `predict` method.
    The wrapper handles image inputs (path / PIL / tensor) and runs model.generate.
    """

    def __init__(self, cfg: FastVLMConfig):
        self.cfg = cfg
        self.model_path = os.path.expanduser(cfg.model_path)
        self.device = torch.device(cfg.device)

        logger.info("Loading FastVLM model from %s onto %s ...", self.model_path, self.device)
        disable_torch_init()

        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path,
            None,
            model_name,
            device=self.cfg.device,
        )

        self.model = self.model.to(self.device)
        self.model = quantize_model_8bit(self.model)
        # safe-guard pad token
        try:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        except Exception:
            pass

        check_layers(self.model)
        logger.info("FastVLM model loaded successfully.")

    def predict(
        self,
        image_input: Any,
        prompt: str,
        conv_mode: str = "qwen_2",
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Inference function that accepts:
        - str : path to image file
        - PIL.Image.Image : already loaded image
        - torch.Tensor : preprocessed tensor from process_images()

        Returns a decoded string (strippped).
        """
        # --- Handle image input ---
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path does not exist: {image_input}")
            logger.info("Inference on image file: %s", image_input)
            image = Image.open(image_input).convert("RGB")
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        elif isinstance(image_input, Image.Image):
            logger.info("Inference on in-memory PIL image.")
            image = image_input.convert("RGB")
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        elif isinstance(image_input, torch.Tensor):
            logger.info("Inference on preprocessed tensor.")
            image_tensor = image_input
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}; expected str, PIL.Image.Image, or torch.Tensor."
            )

        # --- Build multimodal prompt ---
        qs = prompt
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        text_prompt = conv.get_prompt()

        # --- Tokenize ---
        input_ids = tokenizer_image_token(
            text_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        # --- Move image to device ---
        image_tensor = image_tensor.unsqueeze(0).to(self.device, dtype=torch.float16)

        # --- Generate ---
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        logger.info("Model output (truncated): %s", outputs[:400])
        return outputs


# -----------------------
# Video helpers (unchanged)
# -----------------------
def get_video_stats(video_path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": float(fps),
        "frame_count": float(frame_count),
        "duration_sec": float(duration),
        "width": float(width),
        "height": float(height),
    }


def extract_scenes(video_path: str, threshold: float) -> List[Tuple[int, int]]:
    """
    Use PySceneDetect to get scene boundaries.
    Returns list of (start_frame, end_frame).
    """
    start_time = time.perf_counter()
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(video_manager)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for scene in scene_list:
        start = scene[0].get_frames()
        end = scene[1].get_frames()
        scenes.append((start, end))

    video_manager.release()
    logger.info("Detected %d scenes in %.2fs", len(scenes), time.perf_counter() - start_time)
    return scenes


def extract_key_frames(
    video_path: str,
    scenes: List[Tuple[int, int]],
    max_resolution: int,
) -> List[Tuple[float, np.ndarray]]:
    """
    Extract a representative frame per scene, downscaled to max_resolution.
    Returns list of (timestamp_sec, frame_bgr).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames: List[Tuple[float, np.ndarray]] = []

    for (start_frame, end_frame) in scenes:
        mid_frame = int((start_frame + end_frame) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side > max_resolution:
            scale = max_resolution / max_side
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        ts = mid_frame / fps if fps > 0 else 0.0
        frames.append((ts, frame))

    cap.release()
    logger.info("Extracted %d raw key frames", len(frames))
    return frames


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float32)
    b_flat = b.reshape(-1).astype(np.float32)
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
    return float(np.dot(a_flat, b_flat) / denom)


def deduplicate_frames(
    frames: List[Tuple[float, np.ndarray]],
    threshold: float,
) -> List[Tuple[float, np.ndarray]]:
    """
    Simple cosine-similarity based deduplication.
    """
    if not frames:
        return []

    kept: List[Tuple[float, np.ndarray]] = [frames[0]]
    last_vec = frames[0][1]

    for ts, frame in frames[1:]:
        sim = _cosine_similarity(last_vec, frame)
        if sim >= threshold:
            continue
        kept.append((ts, frame))
        last_vec = frame

    logger.info("Deduplicated frames: %d -> %d", len(frames), len(kept))
    return kept

def build_collage(
    frames: List[np.ndarray],
    size: int = 512,
) -> Image.Image:
    """
    Build a 2x2 collage from up to 4 RGB frames.
    Layout:
      [0 | 1]
      [2 | 3]
    """
    imgs = []

    for f in frames[:4]:
        img = Image.fromarray(f)
        img = img.resize((size, size), Image.BILINEAR)
        imgs.append(img)

    # pad if fewer than 4
    while len(imgs) < 4:
        imgs.append(imgs[-1])

    collage = Image.new("RGB", (size * 2, size * 2))
    collage.paste(imgs[0], (0, 0))
    collage.paste(imgs[1], (size, 0))
    collage.paste(imgs[2], (0, size))
    collage.paste(imgs[3], (size, size))

    return collage


# -----------------------
# Small helpers for stage-0
# -----------------------
def sanitize_caption(text: str, max_chars: int = 1024) -> str:
    """Collapse whitespace, replace double quotes, truncate."""
    s = " ".join(text.split())
    s = s.replace('"', "'")
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s

def frame_motion_score(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.resize(a, (64, 64))
    b = cv2.resize(b, (64, 64))
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0
    return float(diff)

def select_motion_keyframes(
    video_path: str,
    scenes: List[Tuple[int, int]],
    stats: Dict[str, float],
    max_resolution: int,
    scene_sample_rate: int,
    motion_threshold: float,
    min_scene_duration_sec: float,
    max_frames: int,
    max_frames_per_scene: int = 2,
) -> List[Tuple[float, np.ndarray]]:
    """
    Motion-prioritized but scene-safe keyframe selection.

    Strategy:
    1. Always keep ONE anchor frame per valid scene (scene midpoint)
    2. Add motion frames on top (ranked by motion)
    3. Preserve temporal order
    4. Enforce global max_frames by dropping lowest-motion frames
    """

    cap = cv2.VideoCapture(video_path)
    fps = stats.get("fps", 1) or 1

    # (timestamp, frame, motion_score)
    selected: List[Tuple[float, np.ndarray, float]] = []

    logger.info(
        "motion_select:start scenes=%d sample_rate=%d motion_th=%.3f max_frames=%d",
        len(scenes),
        scene_sample_rate,
        motion_threshold,
        max_frames,
    )

    for scene_id, (start, end) in enumerate(scenes):
        duration = (end - start) / fps
        if duration < min_scene_duration_sec:
            logger.info(
                "scene %d skipped (duration %.2fs < %.2fs)",
                scene_id,
                duration,
                min_scene_duration_sec,
            )
            continue

        # --------------------------------------------------
        # 1️⃣ Anchor frame (always keep one per scene)
        # --------------------------------------------------
        mid_idx = int((start + end) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ok, anchor = cap.read()

        if ok and anchor is not None:
            ts = mid_idx / fps
            h, w = anchor.shape[:2]
            scale = min(1.0, max_resolution / max(h, w))
            if scale < 1.0:
                anchor = cv2.resize(anchor, (int(w * scale), int(h * scale)))

            selected.append((ts, anchor, 0.0))
            logger.debug("scene %d anchor kept @ %.2fs", scene_id, ts)

        # --------------------------------------------------
        # 2️⃣ Motion frames (additive, not mandatory)
        # --------------------------------------------------
        idxs = np.linspace(start, end - 1, scene_sample_rate, dtype=int)
        prev = None
        scene_motion: List[Tuple[float, np.ndarray, float]] = []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            ts = idx / fps

            h, w = frame.shape[:2]
            scale = min(1.0, max_resolution / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            if prev is None:
                prev = frame
                continue

            motion = frame_motion_score(prev, frame)
            prev = frame

            if motion >= motion_threshold:
                scene_motion.append((ts, frame, motion))

        logger.info(
            "scene %d motion_candidates=%d",
            scene_id,
            len(scene_motion),
        )

        # keep top-N motion frames *per scene*
        scene_motion.sort(key=lambda x: x[2], reverse=True)
        scene_motion = scene_motion[: max_frames_per_scene - 1]

        # restore temporal order
        scene_motion.sort(key=lambda x: x[0])

        selected.extend(scene_motion)

    cap.release()

    logger.info("motion_select:before_global selected=%d", len(selected))

    # --------------------------------------------------
    # 3️⃣ Preserve temporal order globally
    # --------------------------------------------------
    selected.sort(key=lambda x: x[0])

    # --------------------------------------------------
    # 4️⃣ Enforce global frame budget (drop lowest-motion)
    # --------------------------------------------------
    if len(selected) > max_frames:
        ranked = sorted(
            enumerate(selected),
            key=lambda x: x[1][2],  # x[1] = (ts, frame, motion_score)
            reverse=True,
        )


        keep_indices = set(idx for idx, _ in ranked[:max_frames])
        dropped = len(selected) - len(keep_indices)

        selected = [
            frame for idx, frame in enumerate(selected)
            if idx in keep_indices
        ]

        logger.info(
            "motion_select:global_drop dropped=%d kept=%d",
            dropped,
            len(selected),
        )

    logger.info(
        "motion_select:final frames=%d timestamps=%s",
        len(selected),
        [round(ts, 2) for ts, _, _ in selected],
    )

    # return (timestamp, frame)
    return [(ts, frame) for ts, frame, _ in selected]


def build_collages_from_frames(
    frames: List[Tuple[float, np.ndarray]],
    frames_per_collage: int,
    max_collages: int,
) -> List[Dict[str, Any]]:

    collages = []

    for i in range(0, len(frames), frames_per_collage):
        if len(collages) >= max_collages:
            break

        chunk = frames[i : i + frames_per_collage]

        rgb_imgs = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for _, frame in chunk
        ]

        collage_img = build_collage(rgb_imgs)

        collages.append({
            "image": collage_img,
            "start_sec": float(min(ts for ts, _ in chunk)),
            "end_sec": float(max(ts for ts, _ in chunk)),
        })

    return collages


# -----------------------
# Engine facade & result dataclass
# -----------------------
@dataclass
class VideoSummaryResult:
    summary: str
    analysis: Optional[Dict[str, Any]]
    visual_tags: List[str]
    vibe_tags: List[str]
    safety_flags: List[str]
    scenes_detected: int
    frames_used: int
    captions: List[Dict[str, Any]]  # {"timestamp_sec": float, "caption": str}
    timing: Dict[str, float]

    # Stage hooks (placeholders)
    category: Optional[List[str]] = None
    tags: Optional[Dict[str, List[str]]] = None
    diagnostics: Optional[Dict[str, Any]] = None


class FastVLMEngine:
    """
    High-level engine for Stage-0:
      - frame extraction + dedup
      - per-frame human-readable captioning via FastVLM
      - outputs a VideoSummaryResult with captions list

    Stage-1 and Stage-2 hooks are provided as small methods that can be
    implemented and plugged in later.
    """

    def __init__(self, cfg: Optional[FastVLMConfig] = None):
        self.cfg = cfg or load_fastvlm_config()
        logger.setLevel(self.cfg.log_level.upper())
        logger.info("Initializing FastVLMEngine with config: %s", self.cfg)

        self.model = FastVLMModel(self.cfg)

    # -------------------------
    # Stage-1 placeholder: content category classifier
    # -------------------------
    def classify_content_type(self, summary: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Placeholder: return list of category strings (e.g., ["fashion", "lifestyle"]).
        Implement or replace later with zero-shot classifier or small trained model.
        """
        # Default: empty list => uncategorized
        return []

    # -------------------------
    # Stage-2 placeholder: category-specific tagger
    # -------------------------
    def assign_tags(self, summary: str, category: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Placeholder: return { "visual": [...], "vibe": [...] } for given category.
        Implement later using vocabulary lookup, heuristics, or an LLM.
        """
        return {"visual": [], "vibe": []}

    # -------------------------
    # Image description helper (thin wrapper)
    # -------------------------
    def describe_image(self, image_path: str, prompt: str = "Describe the image.") -> str:
        return self.model.predict(image_path, prompt)

    # -------------------------
    # Main Stage-0: summarize_video
    # -------------------------
    def summarize_video(self, video_path: str) -> VideoSummaryResult:
        t0 = time.perf_counter()

        # Basic video stats
        stats = get_video_stats(video_path)
        duration = float(stats.get("duration_sec", 0.0) or 0.0)
        logger.info("Video stats: %s", stats)

        if duration > self.cfg.max_video_seconds:
            raise ValueError(
                f"Video too long: {duration:.1f}s > {self.cfg.max_video_seconds:.1f}s"
            )

        # Fallback config (unchanged)
        fallback_cfg = getattr(self.cfg, "fallback", {})
        fb_enabled = fallback_cfg.get("enabled", True)
        fb_strategy = fallback_cfg.get("strategy", "both")
        fb_min_frames = fallback_cfg.get("min_frames", 2)
        fb_max_frames = fallback_cfg.get("max_frames", 3)
        fb_sec_per_frame = fallback_cfg.get("seconds_per_frame", 20.0)

        # Scene detection
        t_scene_start = time.perf_counter()
        scenes = extract_scenes(video_path, self.cfg.scene_threshold)
        t_scene = time.perf_counter() - t_scene_start

        scenes = scenes or []
        if not scenes and fb_enabled and fb_strategy in ("single_scene", "both"):
            logger.warning("No scenes detected for %s — using fallback full-duration scene", video_path)

        # Ensure fallback scene if needed
        if not scenes and fb_enabled:
            # create single full-duration scene
            scenes = [(0, int(self.cfg.max_frames * 1))]  # small safe fallback

        # Extract key frames
        t_frame_start = time.perf_counter()
        raw_frames = extract_key_frames(video_path, scenes, self.cfg.max_resolution)
        frames = deduplicate_frames(raw_frames, self.cfg.frame_similarity_threshold)
        del raw_frames
        t_frames = time.perf_counter() - t_frame_start

        # If no frames -> try uniform sampling fallback
        if not frames and fb_enabled:
            try:
                target = min(max(1, fb_min_frames), fb_max_frames)
                from .video_utils import sample_uniform_frames, compute_fallback_frame_count
                target = compute_fallback_frame_count(duration, fb_min_frames, fb_max_frames, fb_sec_per_frame)
                frames = sample_uniform_frames(video_path, target_frame_count=target, max_res=self.cfg.max_resolution, stats=stats)
            except Exception:
                logger.exception("Fallback uniform sampling failed")
                frames = []

        # Cap frames
        if len(frames) > self.cfg.max_frames:
            idxs = np.linspace(0, len(frames) - 1, self.cfg.max_frames, dtype=int)
            frames = [frames[i] for i in idxs]

        # If still no frames, return an empty but valid result
        if not frames:
            total = time.perf_counter() - t0
            return VideoSummaryResult(
                summary="",
                analysis=None,
                visual_tags=[],
                vibe_tags=[],
                safety_flags=[],
                scenes_detected=len(scenes),
                frames_used=0,
                captions=[],
                timing={
                    "total_sec": total,
                    "scene_detection_sec": t_scene,
                    "frame_processing_sec": t_frames,
                    "captioning_sec": 0.0,
                    "summary_sec": 0.0,
                },
                category=None,
                tags=None,
                diagnostics={"note": "no_frames"},
            )

        # Caption each frame (Stage-0): call FastVLM with HUMAN_FRAME_PROMPT
        captions: List[Dict[str, Any]] = []
        t_cap_start = time.perf_counter()

        for ts, frame in frames:
            # Convert BGR -> RGB and to PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Build sanitized prompt
            # We supply the short surrounding context (if any) as captions placeholder
            # For stage-0 we prefer short context per frame; keep it minimal.
            prompt_context = ""  # by default, empty; could pass frame-level OCR or overlay text later
            prompt = HUMAN_FRAME_PROMPT.replace("{CAPTIONS}", sanitize_caption(prompt_context))

            # Deterministic decode: temperature=0.0, num_beams=1 (change beams for experimentation)
            try:
                frame_caption = self.model.predict(
                    pil_img,
                    prompt=prompt,
                    conv_mode="qwen_2",
                    temperature=0.0,
                    top_p=None,
                    num_beams=1,
                ).strip()
            except Exception:
                logger.exception("Frame captioning failed for ts=%.2f", ts)
                frame_caption = ""

            captions.append({"timestamp_sec": ts, "caption": frame_caption})

        t_caps = time.perf_counter() - t_cap_start

        # --- Summarization + lightweight analysis (unchanged) ---
        t_sum_start = time.perf_counter()
        caption_texts = [c["caption"] for c in captions]
        # summary: short video-level summary from captions (optional)
        try:
            summary = summarize_captions(self.cfg, caption_texts)
        except Exception:
            logger.exception("Caption summarization failed")
            summary = " ".join(caption_texts)[: self.cfg.max_context_chars]

        analysis = {}
        # analysis = analyze_summary(self.cfg, summary) if summary else None
        visual_tags, vibe_tags, safety_flags = [], [], []
        # visual_tags, vibe_tags, safety_flags = tag_from_text(summary)
        t_sum = time.perf_counter() - t_sum_start

        total = time.perf_counter() - t0

        # Optional Stage-1: classify content type (hook)
        category = []
        # try:
        #     category = self.classify_content_type(summary, metadata=None)
        # except Exception:
        #     logger.exception("Stage-1 classification failed")
        #     category = []

        # Optional Stage-2: assign tags given category (hook)
        tags = {"visual": [], "vibe": []}
        # try:
        #     tags = self.assign_tags(summary, category, metadata=None)
        # except Exception:
        #     logger.exception("Stage-2 tagger failed")
        #     tags = {"visual": [], "vibe": []}

        result = VideoSummaryResult(
            summary=summary,
            analysis=analysis,
            visual_tags=visual_tags,
            vibe_tags=vibe_tags,
            safety_flags=safety_flags,
            scenes_detected=len(scenes),
            frames_used=len(frames),
            captions=captions,
            timing={
                "total_sec": total,
                "scene_detection_sec": t_scene,
                "frame_processing_sec": t_frames,
                "captioning_sec": t_caps,
                "summary_sec": t_sum,
            },
            category=category or None,
            tags=tags or None,
            diagnostics={
                "stage": "stage0_text_only",
                "raw_captions_count": len(captions),
            },
        )

        # Cleanup
        del frames, captions, scenes, caption_texts
        gc.collect()
        if self.cfg.device == "cuda":
            torch.cuda.empty_cache()

        return result


    def summarize_video_collage_old(self, video_path: str) -> VideoSummaryResult:
        t0 = time.perf_counter()

        stats = get_video_stats(video_path)
        duration = stats.get("duration_sec", 0.0)

        if duration > self.cfg.max_video_seconds:
            raise ValueError(
                f"Video too long: {duration:.1f}s > {self.cfg.max_video_seconds:.1f}s"
            )

        # Scene detection
        scenes = extract_scenes(video_path, self.cfg.scene_threshold)

        # Fallback: single scene
        if not scenes:
            scenes = [(0, int(stats["frame_count"]))]

        # Extract frames (reuse existing logic)
        raw_frames = extract_key_frames(
            video_path,
            scenes,
            self.cfg.max_resolution,
        )

        if not raw_frames:
            return VideoSummaryResult(
                summary="",
                analysis=None,
                visual_tags=[],
                vibe_tags=[],
                safety_flags=[],
                scenes_detected=len(scenes),
                frames_used=0,
                captions=[],
                timing={"total_sec": time.perf_counter() - t0},
                diagnostics={"note": "no_frames"},
            )

        # --- Select up to 4 temporally spaced frames ---
        frames_rgb = []
        idxs = np.linspace(0, len(raw_frames) - 1, min(4, len(raw_frames)), dtype=int)

        for i in idxs:
            _, frame_bgr = raw_frames[i]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)

        collage_img = build_collage(frames_rgb)

        # --- Single FastVLM call ---
        try:
            summary = self.model.predict(
                collage_img,
                prompt=COLLAGE_SUMMARY_PROMPT,
                conv_mode="qwen_2",
                temperature=0.0,
                max_new_tokens=96,
            )
        except Exception:
            logger.exception("Collage summarization failed")
            summary = ""

        total = time.perf_counter() - t0

        return VideoSummaryResult(
            summary=summary.strip(),
            analysis=None,
            visual_tags=[],
            vibe_tags=[],
            safety_flags=[],
            scenes_detected=len(scenes),
            frames_used=len(frames_rgb),
            captions=[],  # no per-frame captions in collage mode
            timing={
                "total_sec": total,
            },
            diagnostics={
                "mode": "collage_summary",
                "frames_used": len(frames_rgb),
            },
        )


    def summarize_video_collage(self, video_path: str) -> VideoSummaryResult:
        logger.info("summarize_video_collage:start video=%s", video_path)

        stats_log = {
            "scenes_detected": 0,
            "frames_selected": 0,
            "frames_used": 0,
            "frames_dropped": 0,
            "collages_built": 0,
        }

        t0 = time.perf_counter()
        stats = get_video_stats(video_path)

        logger.info(
            "video_stats fps=%.2f frames=%d duration=%.2fs",
            stats.get("fps", 0),
            stats.get("frame_count", 0),
            stats.get("duration_sec", 0.0),
        )

        # ---------------------------
        # Scene detection
        # ---------------------------
        scenes = extract_scenes(video_path, self.cfg.scene_threshold)
        stats_log["scenes_detected"] = len(scenes)

        logger.info("scenes_detected count=%d scenes=%s", len(scenes), scenes)

        if not scenes:
            scenes = [(0, int(stats["frame_count"]))]
            logger.info("no scenes detected, using fallback full-video scene")

        # ---------------------------
        # Frame budget
        # ---------------------------
        max_frames = self.cfg.max_collages * self.cfg.frames_per_collage
        logger.info(
            "frame_budget max_frames=%d (max_collages=%d × frames_per_collage=%d)",
            max_frames,
            self.cfg.max_collages,
            self.cfg.frames_per_collage,
        )

        # ---------------------------
        # Motion-based frame selection
        # ---------------------------
        frames = select_motion_keyframes(
            video_path=video_path,
            scenes=scenes,
            stats=stats,
            max_resolution=self.cfg.max_resolution,
            scene_sample_rate=self.cfg.scene_sample_rate,
            motion_threshold=self.cfg.motion_threshold,
            min_scene_duration_sec=self.cfg.min_scene_duration_sec,
            max_frames=max_frames,
        )

        stats_log["frames_selected"] = len(frames)

        logger.info(
            "motion_frames_selected count=%d timestamps=%s",
            len(frames),
            [round(ts, 2) for ts, _ in frames],
        )

        if not frames:
            logger.warning("no motion frames selected, returning empty result")
            return VideoSummaryResult(
                summary="",
                analysis=None,
                visual_tags=[],
                vibe_tags=[],
                safety_flags=[],
                scenes_detected=len(scenes),
                frames_used=0,
                captions=[],
                timing={"total_sec": time.perf_counter() - t0},
                diagnostics={"note": "no_motion_frames"},
            )

        # ---------------------------
        # Build collages (order-preserving)
        # ---------------------------
        collages = build_collages_from_frames(
            frames,
            frames_per_collage=self.cfg.frames_per_collage,
            max_collages=self.cfg.max_collages,
        )

        stats_log["collages_built"] = len(collages)
        stats_log["frames_used"] = sum(
            len(range(i * self.cfg.frames_per_collage,
                    min((i + 1) * self.cfg.frames_per_collage, len(frames))))
            for i in range(len(collages))
        )
        stats_log["frames_dropped"] = len(frames) - stats_log["frames_used"]

        logger.info(
            "collages_built=%d frames_used=%d frames_dropped=%d",
            stats_log["collages_built"],
            stats_log["frames_used"],
            stats_log["frames_dropped"],
        )

        # ---------------------------
        # Collage captioning
        # ---------------------------
        captions = []

        for idx, c in enumerate(collages):
            logger.info(
                "collage[%d] start=%.2fs end=%.2fs",
                idx,
                c["start_sec"],
                c["end_sec"],
            )

            try:
                text = self.model.predict(
                    c["image"],
                    prompt=COLLAGE_SUMMARY_PROMPT_v1,
                    conv_mode="qwen_2",
                    temperature=0.0,
                    max_new_tokens=64,
                )

                caption_text = text.strip() if text else ""

                captions.append({
                    "collage_index": idx,
                    "start_sec": c["start_sec"],
                    "end_sec": c["end_sec"],
                    "caption": caption_text,
                })

                logger.info(
                    "collage_caption[%d] %.2f-%.2f: %s",
                    idx,
                    c["start_sec"],
                    c["end_sec"],
                    caption_text,
                )

            except Exception:
                logger.exception("collage inference failed index=%d", idx)

        # ---------------------------
        # Build deterministic summary (timestamp join)
        # ---------------------------
        summary_lines = []
        for c in captions:
            if not c["caption"]:
                continue
            summary_lines.append(
                f"[{round(c['start_sec'],1)}s-{round(c['end_sec'],1)}s] {c['caption']}"
            )

        final_summary = "\n".join(summary_lines)

        logger.info(
            "video_sampling_summary",
            extra={
                "video_path": video_path,
                **stats_log,
                "fastvlm_calls": stats_log["collages_built"],
            },
        )

        logger.info("final_summary:\n%s", final_summary)

        return VideoSummaryResult(
            summary=final_summary,
            analysis=None,
            visual_tags=[],
            vibe_tags=[],
            safety_flags=[],
            scenes_detected=len(scenes),
            frames_used=stats_log["frames_used"],
            captions=captions,
            timing={"total_sec": time.perf_counter() - t0},
            diagnostics={
                "mode": "scene_motion_collage",
                **stats_log,
            },
        )

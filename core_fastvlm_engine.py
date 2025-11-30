# core_fastvlm_engine.py
import os
import time
import json
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
# Quantization helpers
# -----------------------
SKIP_MODULES = ["vision", "clip", "projector", "lm_head", "embed", "norm"]


def quantize_model_8bit(model: torch.nn.Module) -> torch.nn.Module:
    """
    Convert linear layers to 8-bit (except vision & projector etc.).
    """
    import bitsandbytes as bnb
    import torch.nn as nn

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
# FastVLM core model
# -----------------------
class FastVLMModel:
    """
    Thin wrapper around LLaVA/FastVLM that exposes a single `predict` method.
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
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        check_layers(self.model)
        logger.info("FastVLM model loaded successfully.")

    def predict(
        self,
        image_input: Any,
        prompt: str,
        conv_mode: str = "qwen_2",
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        num_beams: int = 1,
    ) -> str:
        """
        Inference function that accepts:
        - str : path to image file
        - PIL.Image.Image : already loaded image
        - torch.Tensor : preprocessed tensor from process_images()
        """
        # --- Handle image input ---
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path does not exist: {image_input}")
            logger.info("Inference on image file: %s", image_input)
            image = Image.open(image_input).convert("RGB")
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_size = image.size

        elif isinstance(image_input, Image.Image):
            logger.info("Inference on in-memory PIL image.")
            image = image_input.convert("RGB")
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_size = image.size

        elif isinstance(image_input, torch.Tensor):
            logger.info("Inference on preprocessed tensor.")
            image_tensor = image_input
            c, h, w = image_tensor.shape[-3:]
            image_size = (w, h)

        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}; expected str, PIL.Image.Image, or torch.Tensor."
            )

        # --- Build multimodal prompt ---
        qs = prompt
        if self.model.config.mm_use_im_start_end:
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
                max_new_tokens=256,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        logger.info("Output: %s", outputs)
        return outputs


# -----------------------
# Video helpers
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
    start_time = time.time()
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
    logger.info("Detected %d scenes in %.2fs", len(scenes), time.time() - start_time)
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


# -----------------------
# Text summarizer/analyzer (lazy)
# -----------------------
_summarizer = None
_analyzer = None


def get_summarizer(device: str):
    global _summarizer
    if _summarizer is not None:
        return _summarizer

    from transformers import pipeline

    logger.info("Loading summarization model (flan-t5-base -> pegasus-xsum fallback) ...")
    try:
        _summarizer = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        )
    except Exception:
        logger.exception("Failed to load flan-t5-base, falling back to pegasus-xsum.")
        _summarizer = pipeline(
            "summarization",
            model="google/pegasus-xsum",
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        )
    return _summarizer


def get_analyzer(device: str):
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    from transformers import pipeline

    logger.info("Loading analyzer model (microsoft/phi-3-mini-4k-instruct) ...")
    _analyzer = pipeline(
        "text-generation",
        model="microsoft/phi-3-mini-4k-instruct",
        device=0 if device == "cuda" and torch.cuda.is_available() else -1,
    )
    return _analyzer


def summarize_captions(
    cfg: FastVLMConfig,
    captions: List[str],
) -> str:
    if not cfg.enable_summary or not captions:
        return " ".join(captions)

    summarizer = get_summarizer(cfg.device)

    joined = " ".join(captions)
    if len(joined) > cfg.max_context_chars:
        joined = joined[: cfg.max_context_chars]

    prompt = f"Summarize what is happening in this short social media video:\n{joined}"
    out = summarizer(prompt, max_new_tokens=64, num_return_sequences=1)[0]
    # flan uses 'generated_text', pegasus uses 'summary_text'
    summary = out.get("generated_text") or out.get("summary_text") or ""
    return summary.strip()


def analyze_summary(cfg: FastVLMConfig, summary: str) -> Optional[Dict[str, Any]]:
    if not cfg.enable_analysis or not summary:
        return None

    analyzer = get_analyzer(cfg.device)

    system_prompt = (
        "You are an assistant that analyzes short social media videos and returns a concise JSON "
        "with keys: 'video_type', 'setting', 'activity', 'people_count', 'vibe'. "
        "No explanation, only JSON."
    )
    user_prompt = f"Video description: {summary}"

    prompt = system_prompt + "\n" + user_prompt
    out = analyzer(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]

    # Try to extract JSON from the output
    try:
        start = out.index("{")
        end = out.rindex("}") + 1
        return json.loads(out[start:end])
    except Exception:
        logger.exception("Failed to parse analyzer output as JSON: %s", out)
        return None


def tag_from_text(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Very dumb heuristic tags for MVP.
    """
    text_l = text.lower()
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

    if any(w in text_l for w in ["tutorial", "how to", "guide", "tips"]):
        vibe_tags.append("educational")
    if any(w in text_l for w in ["motivational", "inspiring", "inspiration"]):
        vibe_tags.append("motivational")
    if any(w in text_l for w in ["funny", "lol", "joke", "meme"]):
        vibe_tags.append("funny")

    # placeholder safety
    if any(w in text_l for w in ["blood", "gun", "fight"]):
        safety_flags.append("violence")

    return visual_tags, vibe_tags, safety_flags


# -----------------------
# Engine facade
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
    captions: List[Dict[str, Any]]
    timing: Dict[str, float]


class FastVLMEngine:
    """
    High-level engine that:
    - wraps the FastVLMModel
    - provides image description
    - provides video summarization with sane caps
    """

    def __init__(self, cfg: Optional[FastVLMConfig] = None):
        self.cfg = cfg or load_fastvlm_config()
        # Align logging level to config
        logger.setLevel(self.cfg.log_level.upper())
        logger.info("Initializing FastVLMEngine with config: %s", self.cfg)

        self.model = FastVLMModel(self.cfg)

    def describe_image(self, image_path: str, prompt: str = "Describe the image.") -> str:
        return self.model.predict(image_path, prompt)

    def summarize_video(self, video_path: str) -> VideoSummaryResult:
        t0 = time.perf_counter()

        stats = get_video_stats(video_path)
        logger.info("Video stats: %s", stats)

        if stats["duration_sec"] > self.cfg.max_video_seconds:
            raise ValueError(
                f"Video too long: {stats['duration_sec']:.1f}s > {self.cfg.max_video_seconds:.1f}s"
            )

        # Scene detection
        t_scene_start = time.perf_counter()
        scenes = extract_scenes(video_path, self.cfg.scene_threshold)
        t_scene = time.perf_counter() - t_scene_start

        # Frame extraction + dedup
        t_frame_start = time.perf_counter()
        raw_frames = extract_key_frames(video_path, scenes, self.cfg.max_resolution)
        frames = deduplicate_frames(raw_frames, self.cfg.frame_similarity_threshold)

        # Cap max frames
        if len(frames) > self.cfg.max_frames:
            idxs = np.linspace(0, len(frames) - 1, self.cfg.max_frames, dtype=int)
            frames = [frames[i] for i in idxs]
        t_frames = time.perf_counter() - t_frame_start

        # Caption frames
        captions: List[Dict[str, Any]] = []
        t_cap_start = time.perf_counter()
        for ts, frame in frames:
            # OpenCV BGR -> PIL RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            caption = self.model.predict(
                pil_img,
                prompt="Describe what is happening in this frame of a short social media video.",
            )
            captions.append(
                {
                    "timestamp_sec": ts,
                    "caption": caption,
                }
            )
        t_caps = time.perf_counter() - t_cap_start

        # Summarize & analyze
        t_sum_start = time.perf_counter()
        caption_texts = [c["caption"] for c in captions]
        summary = summarize_captions(self.cfg, caption_texts)
        analysis = analyze_summary(self.cfg, summary) if summary else None
        visual_tags, vibe_tags, safety_flags = tag_from_text(summary)
        t_sum = time.perf_counter() - t_sum_start

        total = time.perf_counter() - t0

        return VideoSummaryResult(
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
        )

#!/usr/bin/env python3
"""
FastVLM Unified Media Server
----------------------------
Handles both image and video summarization using a local FastVLM model.
Optimized for CUDA inference, scene detection, and contextual summarization.
"""

import os
import cv2
import json
import time
import torch
import numpy as np
from flask import Flask, request, jsonify, Response
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from skimage.metrics import structural_similarity as ssim
from transformers import pipeline
from PIL import Image

# Import your working FastVLM server interface
from fastvlm_server import FastVLMServer
SUMMARIZER_MODE = 'flan'
# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "/home/shang/dev/src/pugsy_ai/pipelines/vlm_pipeline/fastvlm/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3"
SCENE_THRESHOLD = 30.0
FRAME_SIMILARITY_THRESHOLD = 0.90
MAX_CONTEXT_CHARS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)

print("[INFO] Loading FastVLM model onto GPU...")
vlm = FastVLMServer(MODEL_PATH)
print("[INFO] FastVLM ready.")



# print("[INFO] Loading summarization model (google/pegasus-xsum)...")
# summarizer = pipeline("summarization", model="google/pegasus-xsum", device=0 if DEVICE == "cuda" else -1)
# print("[INFO] Summarizer ready.")

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

print("[INFO] Loading summarization model (prefer flan-t5-base, fallback pegasus-xsum)...")
try:
    # flan-t5-large follows instructions well for "understanding"-style prompts
    summarizer = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0 if DEVICE == "cuda" else -1,
        truncation=True,
    )
    SUMMARIZER_MODE = "flan"
    print("[INFO] Summarizer ready: google/flan-t5-base")
except Exception:
    summarizer = pipeline("summarization", model="google/pegasus-xsum", device=0 if DEVICE == "cuda" else -1)
    SUMMARIZER_MODE = "pegasus"
    print("[INFO] Summarizer ready: google/pegasus-xsum (fallback)")


# Warm-up CUDA kernels
# print("[INFO] Warming up CUDA kernels...")
# # Create a simple gray image for warm-up
# dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
# _ = vlm.predict(dummy_image, "Describe the image.")
# torch.cuda.synchronize()
# print("[INFO] Warm-up complete.")

# Warm-up CUDA kernels
print("[INFO] Warming up CUDA kernels...")
dummy_image = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
_ = vlm.predict(dummy_image, "Describe the image.")
torch.cuda.synchronize()
print("[INFO] Warm-up complete.")

try:
    print("[INFO] Loading semantic analyzer (microsoft/phi-3-mini-4k-instruct)...")
    from transformers import pipeline
    phi3_analyzer = pipeline(
        "text-generation",
        model="microsoft/phi-3-mini-4k-instruct",
        torch_dtype=torch.bfloat16,
        device=0 if DEVICE == "cuda" else -1
    )
    print("[INFO] Analyzer ready.")
except Exception as e:
    print("[ERORR]: Failed to initialize microsoft/phi-3-mini-4k-instruct")
# -----------------------------
# Helper functions
# -----------------------------
def extract_scenes(video_path, threshold=SCENE_THRESHOLD):
    """Detect scene boundaries using PySceneDetect."""
    start = time.time()
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()
    duration = time.time() - start
    print(f"[INFO] Detected {len(scenes)} scenes in {duration:.2f}s")
    return scenes, duration


def extract_key_frames(video_path, scenes):
    """Extract a representative mid-frame from each scene."""
    start = time.time()
    cap = cv2.VideoCapture(video_path)
    frames = []

    for i, (start_time, end_time) in enumerate(scenes):
        mid_frame_time = (start_time.get_seconds() + end_time.get_seconds()) / 2.0
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_frame_time * 1000)
        success, frame = cap.read()
        if success:
            frames.append(frame)
        else:
            print(f"[WARN] Could not read frame for scene {i}")

    cap.release()
    duration = time.time() - start
    print(f"[INFO] Extracted {len(frames)} frames in {duration:.2f}s")
    return frames, duration


def frame_similarity(frame_a, frame_b):
    """Compute SSIM similarity between two frames."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.resize(gray_a, (320, 320))
    gray_b = cv2.resize(gray_b, (320, 320))
    score, _ = ssim(gray_a, gray_b, full=True)
    return score


def deduplicate_frames(frames, threshold=FRAME_SIMILARITY_THRESHOLD):
    """Remove visually redundant frames."""
    if not frames:
        return []
    unique = [frames[0]]
    for f in frames[1:]:
        if frame_similarity(unique[-1], f) < threshold:
            unique.append(f)
    print(f"[INFO] Deduplicated frames: {len(frames)} → {len(unique)}")
    return unique


# def build_prompt(previous_context):
#     """
#         Optional specialized variants

#     You can dynamically adjust based on category if you know the video type:

#     Makeup/Fashion

#     "Describe what the person is doing in this frame — note their makeup, clothing, gestures, and tools or products visible. Identify any aesthetic style or trend that stands out."

#     Cooking/Food

#     "Describe the main action in this frame — what ingredients, tools, or cooking steps are visible. Mention colors, textures, and presentation details."

#     Travel/Vlog

#     "Describe what’s happening, where it might be taking place, and the overall tone — e.g. relaxed, adventurous, cinematic, or casual."

#     You can programmatically swap these in based on the IG post metadata or hashtags.
#     """
#     """Create natural prompt for model."""
#     instruction_bk = (
#         "You are a concise visual narrator. "
#         "Describe exactly what is visible in the image in one or two short sentences. "
#         "Be factual and avoid speculation."
#     )
#     instruction = (
#         """You are a perceptive visual analyst. Describe what is happening in this frame, focusing on the main subject, their actions, expressions, style, and any visible text or objects. 
#         Include details that reveal the context or mood — for example, whether it seems like a tutorial, performance, vlog, or product showcase. 
#         Be concise, objective, and avoid guessing unseen details.""")
#     if previous_context:
#         return f"{instruction}\nPrevious context: {previous_context}\nNow describe this scene:"
#     return f"{instruction}\nDescribe this scene:"


def build_prompt(previous_context: str = "") -> str:
    """
    Build a context-aware captioning prompt for FastVLM.
    Encourages concise factual description + contextual hints (action, style, visible text).
    """
    base = (
        "You are a concise visual analyst. For the provided frame, describe the main subject, "
        "what they are doing (action), any visible tools/objects, visible text, and the overall mood or style. "
        "Keep it factual and brief (1-2 sentences). Avoid guessing facts not shown in the frame."
    )
    if previous_context:
        return f"{base}\nPrevious context: {previous_context}\nNow describe this scene:"
    return base + "\nDescribe this scene:"



def summarize_text(captions):
    """
    Produce a single coherent summary from per-scene captions.
    Works with either FLAN (instruction-following) or Pegasus (abstractive).
    """
    if not captions:
        return ""

    # 1) Remove exact/repetitive duplicates (preserve order)
    unique = []
    for c in captions:
        txt = c.strip()
        if not txt:
            continue
        if not unique or txt.lower() != unique[-1].lower():
            unique.append(txt)

    if not unique:
        return ""

    # 2) If there is only one unique caption, return it (or slightly rephrase with summarizer)
    joined = " Then, ".join(unique)

    # 3) Heuristic to set lengths based on input token count
    input_words = len(joined.split())
    # for short inputs keep it concise
    max_len = max(20, int(input_words * 0.6) + 10)
    min_len = max(8, int(input_words * 0.25))

    # 4) Use the chosen summarizer
    if SUMMARIZER_MODE == "flan":
        prompt = (
            "You are given short scene descriptions from an Instagram-style short video. "
            "Write a single-paragraph summary that captures: the main subject (who/what), the key actions, "
            "the visual style or mood, and the likely intent (tutorial, performance, demo, ad), if visible. "
            "Be factual and concise.\n\n"
            f"Scenes:\n{joined}"
        )
        out = summarizer(prompt, max_length=max_len, min_length=min_len, do_sample=False)
        # flan returns a list of dicts, value key may be 'generated_text' or 'text'
        text = out[0].get("generated_text") or out[0].get("text") or out[0].get("summary_text") or out[0].get("output_text")
        return text.strip()
    else:
        # Pegasus expects a plain text input without instruction prefix
        out = summarizer(joined, max_length=max_len, min_length=min_len, do_sample=False)
        return out[0]["summary_text"].strip()


def analyze_summary(summary_text):
    """
    Convert a textual video summary into structured analysis:
      - subject
      - action
      - tone / mood
      - likely_intent
    Uses Phi-3-mini (small, local, instruction-tuned).
    """
    if not summary_text:
        return {}

    from transformers import pipeline
    import json, re, torch

    device = 0 if torch.cuda.is_available() else -1
    # model_name = "microsoft/phi-3-mini-4k-instruct"
    # extractor = pipeline(
    #     "text-generation",
    #     model=model_name,
    #     torch_dtype=torch.bfloat16,
    #     device=device
    # )

    prompt = (
        "You are a structured video content analyst. "
        "From the summary below, extract and return ONLY valid JSON with the keys: "
        "subject, action, tone, likely_intent. "
        "Fill every field concisely based on what the summary describes. "
        "No explanations, just one JSON object.\n\n"
        f"Summary: \"{summary_text}\"\n\n"
        "JSON:"
    )

    try:
        out = phi3_analyzer(prompt, max_new_tokens=160, do_sample=False)[0]["generated_text"]

        # Clean and parse JSON
        json_str = re.search(r"\{.*\}", out, re.DOTALL)
        if json_str:
            return json.loads(json_str.group(0))
        else:
            return {"raw": out.strip()}
    except Exception as e:
        return {"error": str(e), "raw_output": out}


import re
import json

# -----------------------------
# Inference functions
# -----------------------------
def generate_image_caption(image_path, prompt="Describe the image."):
    """Single image inference."""
    start = time.time()
    output = vlm.predict(image_path, prompt)
    elapsed = time.time() - start
    return {"caption": output.strip(), "timing_sec": round(elapsed, 2)}


def generate_video_summary(video_path, scene_threshold=SCENE_THRESHOLD):
    """Video summarization with streaming-friendly structure."""
    total_start = time.time()

    scenes, t_scene = extract_scenes(video_path, scene_threshold)
    frames, t_extract = extract_key_frames(video_path, scenes)
    frames = deduplicate_frames(frames)

    captions = []
    prev_context = ""
    t_caption_start = time.time()

    for i, frame in enumerate(frames):
        start = time.time()
        # Convert BGR -> RGB PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        caption_prompt = build_prompt(prev_context)
        caption = vlm.predict(pil_image, caption_prompt).strip()
        elapsed = time.time() - start
        captions.append(caption)
        prev_context = " ".join(captions[-2:])[-MAX_CONTEXT_CHARS:]
        print(f"[INFO] Scene {i+1}/{len(frames)} → {caption} ({elapsed:.2f}s)")


    t_caption = time.time() - t_caption_start
    summary = summarize_text(captions)
    analysis = analyze_summary(summary)
    total_time = time.time() - total_start

    return {
        "summary": summary,
        "analysis": analysis,
        "scenes_detected": len(scenes),
        "frames_used": len(frames),
        "captions": captions,
        "timing": {
            "scene_detection_sec": round(t_scene, 2),
            "frame_extraction_sec": round(t_extract, 2),
            "captioning_sec": round(t_caption, 2),
            "total_sec": round(total_time, 2),
        },
    }


# -----------------------------
# Flask routes
# -----------------------------
@app.route("/predict_image", methods=["POST"])
def predict_image_endpoint():
    """Handle single image inference."""
    try:
        data = request.get_json()
        image_path = data.get("image_path")
        prompt = data.get("prompt", "Describe the image.")
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": f"Invalid image path: {image_path}"}), 400

        result = generate_image_caption(image_path, prompt)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/summarize_video", methods=["POST"])
def summarize_video_stream():
    """Stream per-scene captions as JSON lines for real-time feedback."""
    try:
        data = request.get_json()
        video_path = data.get("video_path")
        scene_threshold = float(data.get("scene_threshold", SCENE_THRESHOLD))
        if not os.path.exists(video_path):
            return Response(json.dumps({"error": "Invalid video path"}), mimetype="application/json")

        def generate():
            yield json.dumps({"status": "processing", "step": "scene_detection"}) + "\n"
            result = generate_video_summary(video_path, scene_threshold)
            yield json.dumps(result) + "\n"

        return Response(generate(), mimetype="application/json")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)


# 4. Optional “understanding” enhancement

# To push this into a semantic analysis layer, add another step (after summarization) where you prompt a language model (even a small one locally) to extract key dimensions like:

# Subject(s)

# Activity

# Visual setting

# Emotional tone

# Purpose (tutorial, entertainment, ad, etc.)

# Something like:

# analysis_prompt = f"""
# Video summary: {summary}
# From this, extract:
# - Main subject or actor
# - Key action or theme
# - Emotional tone or aesthetic
# - Possible intent of the video
# Give short bullet points.
# """


# 3. If you want richer understanding summaries (semantic fusion)

# Pegasus-XSum gives short factual compression.
# If you want deeper “insight” (e.g., ‘the creator is showing a Halloween makeup transformation’), it’s better to swap Pegasus for a small instruction-tuned summarizer trained for reasoning.

# Try:summarizer = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-large",
#     device=0 if DEVICE == "cuda" else -1
# )
# prompt = (
#     "These are short scene descriptions from an Instagram video. "
#     "Summarize what the video is about, who or what it focuses on, and the overall intent or theme."
#     f"\n\nScenes:\n{joined}"
# )
# result = summarizer(prompt, max_length=120, min_length=40, do_sample=False)
# FLAN-T5 actually follows instructions and will generate context-aware summaries like:

# “The video shows a woman transforming herself into a ‘Mad Hatter’ look for Halloween using white face paint and bright colors.”

# That’s understanding, not just compression.
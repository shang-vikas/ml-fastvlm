#!/usr/bin/env python3
"""
Smart streaming video-to-text summarizer using FastVLM and PySceneDetect.
Streams per-scene captions and final summary incrementally as JSON lines (SSE-like).
"""

import os
import cv2
import json
import time
import torch
import numpy as np
from flask import Flask, Response, request
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from skimage.metrics import structural_similarity as ssim
from transformers import pipeline

# Your FastVLM model interface
from fastvlm_server import FastVLMServer

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "/home/shang/dev/src/pugsy_ai/pipelines/vlm_pipeline/fastvlm/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3"
SCENE_THRESHOLD = 30.0
MAX_CONTEXT_CHARS = 200
FRAME_SIMILARITY_THRESHOLD = 0.90
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__)

print("[INFO] Loading FastVLM model...")
vlm_server = FastVLMServer(MODEL_PATH)
print("[INFO] Model ready.")

print("[INFO] Loading text summarizer (BART)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if DEVICE == "cuda" else -1)
print("[INFO] Summarizer ready.")

# ----------------------------
# Helper functions
# ----------------------------

def extract_scenes(video_path, threshold):
    start = time.time()
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()
    return scenes, time.time() - start


def extract_key_frames(video_path, scenes):
    start = time.time()
    cap = cv2.VideoCapture(video_path)
    frames = []
    for (start_time, end_time) in scenes:
        mid_t = (start_time.get_seconds() + end_time.get_seconds()) / 2.0
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_t * 1000)
        success, frame = cap.read()
        if success:
            frames.append(frame)
    cap.release()
    return frames, time.time() - start


def frame_similarity(a, b):
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.resize(gray_a, (320, 320))
    gray_b = cv2.resize(gray_b, (320, 320))
    score, _ = ssim(gray_a, gray_b, full=True)
    return score


def deduplicate_frames(frames, threshold=FRAME_SIMILARITY_THRESHOLD):
    if not frames:
        return []
    unique = [frames[0]]
    for f in frames[1:]:
        if frame_similarity(unique[-1], f) < threshold:
            unique.append(f)
    return unique


def build_prompt(previous_context):
    instruction = (
        "You are a concise video narrator. "
        "Describe only what is visible in the frame in one or two short sentences. "
        "Be factual and avoid repetition or speculation."
    )
    if previous_context:
        return f"{instruction}\nPrevious scene summary: {previous_context}\nNow describe the current scene:"
    return f"{instruction}\nDescribe the current scene:"


def generate_caption(image, prev_context):
    prompt = build_prompt(prev_context)
    caption = vlm_server.predict(image, prompt).strip()
    return caption


def summarize_text(captions):
    text = " ".join(captions)
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    return summary


# ----------------------------
# Core streaming generator
# ----------------------------
def stream_video_summary(video_path, scene_threshold=SCENE_THRESHOLD):
    if not os.path.exists(video_path):
        yield json.dumps({"error": f"Video not found: {video_path}"}) + "\n"
        return

    total_start = time.time()
    yield json.dumps({"status": "processing", "step": "scene_detection"}) + "\n"

    scenes, t_scene = extract_scenes(video_path, scene_threshold)
    yield json.dumps({"status": "scene_detection_done", "scenes_detected": len(scenes)}) + "\n"

    frames, t_extract = extract_key_frames(video_path, scenes)
    frames = deduplicate_frames(frames)
    yield json.dumps({"status": "frame_extraction_done", "frames_used": len(frames)}) + "\n"

    captions = []
    prev_context = ""
    t_caption_start = time.time()

    for i, frame in enumerate(frames):
        start_time = time.time()
        caption = generate_caption(frame, prev_context)
        elapsed = time.time() - start_time
        captions.append(caption)
        prev_context = " ".join(captions[-2:])[-MAX_CONTEXT_CHARS:]

        yield json.dumps({
            "scene_index": i + 1,
            "caption": caption,
            "elapsed_sec": round(elapsed, 2),
            "status": "scene_caption_done"
        }) + "\n"

    t_caption = time.time() - t_caption_start

    yield json.dumps({"status": "summarizing"}) + "\n"
    summary = summarize_text(captions)

    total_time = time.time() - total_start
    result = {
        "status": "complete",
        "summary": summary,
        "scenes_detected": len(scenes),
        "frames_used": len(frames),
        "captions": captions,
        "timing": {
            "scene_detection_sec": round(t_scene, 2),
            "frame_extraction_sec": round(t_extract, 2),
            "captioning_sec": round(t_caption, 2),
            "total_sec": round(total_time, 2)
        }
    }

    yield json.dumps(result) + "\n"


# ----------------------------
# Flask route (streaming)
# ----------------------------
@app.route("/summarize_video", methods=["POST"])
def summarize_video_stream_endpoint():
    try:
        data = request.get_json()
        video_path = data.get("video_path")
        scene_threshold = float(data.get("scene_threshold", SCENE_THRESHOLD))

        return Response(
            stream_video_summary(video_path, scene_threshold),
            mimetype="application/json"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)


# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, threaded=True)

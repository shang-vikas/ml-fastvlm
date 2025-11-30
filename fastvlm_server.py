# fastvlm_server.py
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request

from .core_fastvlm_engine import FastVLMEngine
from .fastvlm_config import load_fastvlm_config

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=os.getenv("FASTVLM_LOG_LEVEL", "INFO"),
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FastVLMHTTP")


# -----------------------
# App + Engine
# -----------------------
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=int(os.getenv("FASTVLM_WORKERS", "1")))

cfg = load_fastvlm_config()
engine = FastVLMEngine(cfg)


def error_response(message: str, status: int = 400, code: str = "bad_request"):
    return jsonify({"error": {"message": message, "code": code}}), status


# -----------------------
# Health endpoints
# -----------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/readyz", methods=["GET"])
def readyz():
    try:
        # simple readiness check
        _ = engine.model is not None
        return jsonify({"status": "ready"}), 200
    except Exception as e:
        logger.exception("Readiness check failed")
        return jsonify({"status": "error", "error": str(e)}), 500


# -----------------------
# Image prediction
# -----------------------
@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return error_response("Invalid JSON body", 400)

    image_path = data.get("image_path")
    prompt = data.get("prompt", "Describe the image.")

    if not image_path:
        return error_response("Missing 'image_path'", 400)

    if not os.path.exists(image_path):
        return error_response(f"Image not found at path: {image_path}", 400)

    logger.info("predict_image called for %s", image_path)

    future = executor.submit(engine.describe_image, image_path, prompt)
    try:
        output = future.result()
    except FileNotFoundError as e:
        logger.exception("File not found error in predict_image")
        return error_response(str(e), 400)
    except Exception as e:
        logger.exception("Unhandled error in predict_image")
        return error_response(str(e), 500, code="internal_error")

    return jsonify(
        {
            "image_path": image_path,
            "prompt": prompt,
            "output": output,
        }
    )


# -----------------------
# Video summarization
# -----------------------
@app.route("/summarize_video", methods=["POST"])
def summarize_video():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return error_response("Invalid JSON body", 400)

    video_path = data.get("video_path")
    if not video_path:
        return error_response("Missing 'video_path'", 400)

    if not os.path.exists(video_path):
        return error_response(f"Video not found at path: {video_path}", 400)

    logger.info("summarize_video called for %s", video_path)

    # NOTE: You can add per-request overrides for cfg here if you really want,
    # but for MVP we just use the engine config.
    future = executor.submit(engine.summarize_video, video_path)

    try:
        result = future.result()
    except ValueError as e:
        # e.g., video too long, invalid
        logger.exception("Validation error in summarize_video")
        return error_response(str(e), 400)
    except Exception as e:
        logger.exception("Unhandled error in summarize_video")
        return error_response(str(e), 500, code="internal_error")

    return jsonify(
        {
            "summary": result.summary,
            "analysis": result.analysis,
            "visual_tags": result.visual_tags,
            "vibe_tags": result.vibe_tags,
            "safety_flags": result.safety_flags,
            "scenes_detected": result.scenes_detected,
            "frames_used": result.frames_used,
            "captions": result.captions,
            "timing": result.timing,
        }
    )


if __name__ == "__main__":
    # Dev-only. In prod: gunicorn -w 1 -b 0.0.0.0:7860 fastvlm_server:app
    port = int(os.getenv("FASTVLM_PORT", "7860"))
    logger.info("Starting FastVLM HTTP server on 0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)

# fastvlm_server.py
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request

from .core_fastvlm_engine import FastVLMEngine
from .fastvlm_config import load_fastvlm_config
from .tmp_media import TempMedia
from .stage1_classifier import classify_content_type
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
# @app.route("/predict_image", methods=["POST"])
# def predict_image():
#     try:
#         data = request.get_json(force=True) or {}
#     except Exception:
#         return error_response("Invalid JSON body", 400)

#     image_path = data.get("image_path")
#     prompt = data.get("prompt", "Describe the image.")

#     if not image_path:
#         return error_response("Missing 'image_path'", 400)

#     logger.info("predict_image called for %s", image_path)
#     try:
#         with TempMedia(image_path) as local_path:
#             future = executor.submit(engine.describe_image, local_path, prompt)
#             output = future.result()
#             # ---------------------------
#             # Stage-1: content classification (multi-label + scores)
#             # ---------------------------
#             try:
#                 # We classify on the aggregated textual summary produced by FastVLM
#                 # Optionally you can pass the joined captions or creator metadata instead.
#                 summary_text = output or ""
#                 # Use multi_label + return_scores so we can return [(label, score), ...]
#                 stage1_out = classify_content_type(
#                     summary_text,
#                     top_k=4,
#                     use_zero_shot=True,
#                     score_threshold=0.20,
#                     multi_label=True,
#                     return_scores=True,
#                 )
#                 # stage1_out is a list of (label, score) tuples
#                 stage1_labels = [t[0] for t in stage1_out]
#                 stage1_scores = [t[1] for t in stage1_out]
#                 stage1_diagnostics = {"summary_used": summary_text, "raw": stage1_out}
#                 logger.info("Stage-1 classification: %s", stage1_out)
#             except Exception as e:
#                 logger.exception("Stage-1 classifier failed")
#                 stage1_labels = []
#                 stage1_scores = []
#                 stage1_diagnostics = {"error": str(e)}

#             # Use the structured result produced inside summarize_video
#             fastvlm_obj = getattr(output, "fastvlm", None)
#             fastvlm_diag = getattr(output, "fastvlm_diagnostics", None)

#             if fastvlm_obj is not None:
#                 status = 200
#             else:
#                 # structured generation failed/was not available
#                 status = 502

#             payload = {
#                 "fastvlm": fastvlm_obj,
#                 "diagnostics": fastvlm_diag,
#                 "summary": output.summary,
#                 "visual_tags": output.visual_tags,
#                 "vibe_tags": output.vibe_tags,
#                 "safety_flags": output.safety_flags,
#                 "scenes_detected": output.scenes_detected,
#                 "frames_used": output.frames_used,
#                 "captions": output.captions,
#                 "timing": output.timing,
#                 # Stage-1 outputs
#                 "stage1": {
#                     "labels": stage1_labels,
#                     "scores": stage1_scores,
#                 },
#                 "stage1_diagnostics": stage1_diagnostics,
#             }

#             return jsonify(payload), status
#     except ValueError as e:
#         # e.g. download too large
#         logger.exception("Validation/download error in predict_image")
#         return error_response(str(e), 400)
#     except RuntimeError as e:
#         # missing deps or config
#         logger.exception("Runtime error in predict_image")
#         return error_response(str(e), 500)
#     except FileNotFoundError as e:
#         logger.exception("File not found error in predict_image")
#         return error_response(str(e), 400)
#     except Exception as e:
#         logger.exception("Unhandled error in predict_image")
#         return error_response(str(e), 500, code="internal_error")

#     return jsonify(
#         {
#             "image_path": image_path,
#             "prompt": prompt,
#             "output": output,
#         }
#     )

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

    logger.info("predict_image called for %s", image_path)
    try:
        with TempMedia(image_path) as local_path:
            # Run image description (returns a string)
            future = executor.submit(engine.describe_image, local_path, prompt)
            output_text = future.result()

            # ---------------------------
            # Stage-1: content classification (multi-label + scores)
            # ---------------------------
            try:
                summary_text = output_text or ""
                stage1_out = classify_content_type(
                    summary_text,
                    top_k=4,
                    use_zero_shot=True,
                    score_threshold=0.20,
                    multi_label=True,
                    return_scores=True,
                )
                stage1_labels = [t[0] for t in stage1_out]
                stage1_scores = [t[1] for t in stage1_out]
                stage1_diagnostics = {"summary_used": summary_text, "raw": stage1_out}
                logger.info("Stage-1 classification: %s", stage1_out)
            except Exception as e:
                logger.exception("Stage-1 classifier failed")
                stage1_labels = []
                stage1_scores = []
                stage1_diagnostics = {"error": str(e)}

            # Build a clean payload for image prediction
            payload = {
                "image_path": image_path,
                "prompt": prompt,
                "summary": output_text,                       # textual description from FastVLM
                "stage1": {
                    "labels": stage1_labels,
                    "scores": stage1_scores,
                },
                "stage1_diagnostics": stage1_diagnostics,
            }

            return jsonify(payload), 200

    except ValueError as e:
        # e.g. download too large
        logger.exception("Validation/download error in predict_image")
        return error_response(str(e), 400)
    except RuntimeError as e:
        # missing deps or config
        logger.exception("Runtime error in predict_image")
        return error_response(str(e), 500)
    except FileNotFoundError as e:
        logger.exception("File not found error in predict_image")
        return error_response(str(e), 400)
    except Exception as e:
        logger.exception("Unhandled error in predict_image")
        return error_response(str(e), 500, code="internal_error")


# -----------------------
# Video summarization (Stage-0)
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

    logger.info("summarize_video called for %s", video_path)

    try:
        with TempMedia(video_path) as local_path:
            # future = executor.submit(engine.summarize_video, local_path)
            future = executor.submit(engine.summarize_video_collage, local_path)
            video_result = future.result()
            # ---------------------------
            # Stage-1: content classification (multi-label + scores)
            # ---------------------------
            try:
                # We classify on the aggregated textual summary produced by FastVLM
                # Optionally you can pass the joined captions or creator metadata instead.
                summary_text = video_result.summary or ""
                # Use multi_label + return_scores so we can return [(label, score), ...]
                stage1_out = classify_content_type(
                    summary_text,
                    top_k=4,
                    use_zero_shot=True,
                    score_threshold=0.20,
                    multi_label=True,
                    return_scores=True,
                )
                # stage1_out is a list of (label, score) tuples
                stage1_labels = [t[0] for t in stage1_out]
                stage1_scores = [t[1] for t in stage1_out]
                stage1_diagnostics = {"summary_used": summary_text, "raw": stage1_out}
                logger.info("Stage-1 classification: %s", stage1_out)
            except Exception as e:
                logger.exception("Stage-1 classifier failed")
                stage1_labels = []
                stage1_scores = []
                stage1_diagnostics = {"error": str(e)}

            # Use the structured result produced inside summarize_video
            fastvlm_obj = getattr(video_result, "fastvlm", None)
            fastvlm_diag = getattr(video_result, "fastvlm_diagnostics", None)

            if fastvlm_obj is not None:
                status = 200
            else:
                # structured generation failed/was not available
                status = 200

            payload = {
                "fastvlm": fastvlm_obj,
                "diagnostics": fastvlm_diag,
                "summary": video_result.summary,
                "visual_tags": video_result.visual_tags,
                "vibe_tags": video_result.vibe_tags,
                "safety_flags": video_result.safety_flags,
                "scenes_detected": video_result.scenes_detected,
                "frames_used": video_result.frames_used,
                "captions": video_result.captions,
                "timing": video_result.timing,
                # Stage-1 outputs
                "stage1": {
                    "labels": stage1_labels,
                    "scores": stage1_scores,
                },
                "stage1_diagnostics": stage1_diagnostics,
            }

            return jsonify(payload), status
    except ValueError as e:
        logger.exception("Validation/download error in summarize_video")
        return error_response(str(e), 400)
    except RuntimeError as e:
        logger.exception("Runtime error in summarize_video")
        return error_response(str(e), 500)
    except Exception as e:
        logger.exception("Unhandled error in summarize_video")
        return error_response(str(e), 500, code="internal_error")

    # # Compose a stable payload for clients. Keep HTTP 200 for success,
    # # but include a flag that says whether structured output was produced.
    # fastvlm_obj = getattr(video_result, "fastvlm", None)
    # diagnostics = getattr(video_result, "fastvlm_diagnostics", None)

    # payload = {
    #     "structured_available": bool(fastvlm_obj),
    #     "fastvlm": fastvlm_obj,            # may be None during Stage-0
    #     "diagnostics": diagnostics,         # may be None
    #     "summary": video_result.summary,
    #     "visual_tags": video_result.visual_tags,
    #     "vibe_tags": video_result.vibe_tags,
    #     "safety_flags": video_result.safety_flags,
    #     "scenes_detected": video_result.scenes_detected,
    #     "frames_used": video_result.frames_used,
    #     "captions": video_result.captions,
    #     "timing": video_result.timing,
    #     # stage hooks
    #     "category": getattr(video_result, "category", None),
    #     "tags": getattr(video_result, "tags", None),
    # }

    # return jsonify(payload), 200


if __name__ == "__main__":
    # Dev-only. In prod: gunicorn -w 1 -b 0.0.0.0:7860 fastvlm_server:app
    port = int(os.getenv("FASTVLM_PORT", "7860"))
    logger.info("Starting FastVLM HTTP server on 0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)

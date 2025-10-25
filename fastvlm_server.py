import os
import logging
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import bitsandbytes as bnb
from concurrent.futures import ThreadPoolExecutor

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------
MODEL_PATH = "/home/shang/dev/src/pugsy_ai/pipelines/vlm_pipeline/fastvlm/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3"
DEVICE = "cuda"
SKIP_MODULES = ["vision", "clip", "projector", "lm_head", "embed", "norm"]

# -----------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FastVLMServer")


# -----------------------------------------------------
# QUANTIZATION FUNCTION (SKIPS VISION & PROJECTOR)
# -----------------------------------------------------
def quantize_model_8bit(model):
    for name, module in model.named_children():
        lname = name.lower()

        # Skip quantization for visual and projector components
        if any(skip in lname for skip in SKIP_MODULES):
            logger.info(f"Skipping quantization for {name}")
            continue

        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            device = next(module.parameters()).device
            logger.info(f"Quantizing {name} -> Linear8bitLt on {device}")
            qlayer = bnb.nn.Linear8bitLt(
                in_features, out_features, bias=bias, device=device
            )
            setattr(model, name, qlayer)
        else:
            if len(list(module.children())) > 0:
                quantize_model_8bit(module)
    return model


# -----------------------------------------------------
# AUDIT & FIX FUNCTION
# -----------------------------------------------------
def audit_and_move_to_cuda(model, dst_device="cuda"):
    cpu_tensors = []
    for n, p in model.named_parameters():
        if p.device.type != "cuda":
            cpu_tensors.append(n)
    for n, b in model.named_buffers():
        if b.device.type != "cuda":
            cpu_tensors.append(n)

    if cpu_tensors:
        logger.warning(f"Found {len(cpu_tensors)} CPU tensors; moving to CUDA.")
        model.to(dst_device)
    else:
        logger.info("All parameters already on CUDA.")
    return model

def check_layers(model):
    for name, param in list(model.named_parameters())[:50]:
        if any(k in name for k in ["embed", "lm_head", "norm"]):
            logger.info(f"{name} -> {param.dtype}")

# -----------------------------------------------------
# FASTVLM SERVER CLASS
# -----------------------------------------------------
class FastVLMServer:
    def __init__(self, model_path: str):
        self.model_path = os.path.expanduser(model_path)
        self.device = torch.device(DEVICE)

        logger.info(f"Loading FastVLM model from {self.model_path} onto {self.device} ...")
        disable_torch_init()

        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, None, model_name, device="cuda"
        )

        # Move model fully to CUDA before quantization
        self.model = self.model.to(self.device)

        # Quantize (skip vision + projector for stability)
        # self.model = quantize_model_8bit(self.model)

        # Ensure everything now on CUDA
        self.model = audit_and_move_to_cuda(self.model)

        # Convert to FP16 for non-quantized parts
        # self.model = self.model.half().eval()
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                module.half()
        self.model.eval()

        # Optional: compile safely
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"torch.compile skipped due to: {e}")

        # Config
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        logger.info("Model loaded successfully.")

        # Warm-up CUDA kernels
        # self._warmup_model()
        check_layers(self.model)

    def _warmup_model(self):
        logger.info("Warming up CUDA kernels ...")
        dummy_image = torch.zeros((1, 3, 224, 224), dtype=torch.float16, device=self.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            try:
                if hasattr(self.model, "get_model") and hasattr(self.model.get_model(), "get_vision_tower"):
                    vt = self.model.get_model().get_vision_tower()
                    _ = vt(dummy_image)
                else:
                    _ = self.model(dummy_image)
                torch.cuda.synchronize()
                logger.info("Warm-up complete. Model ready.")
            except Exception as e:
                logger.warning(f"Warm-up failed (non-fatal): {e}")

    # -------------------------------------------------
    # INFERENCE FUNCTION
    # -------------------------------------------------

    def predict_bk(self, image_path: str, prompt: str,
                conv_mode: str = "qwen_2",
                temperature: float = 0.2,
                top_p: float = None,
                num_beams: int = 1):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Inference on image: {image_path}")

        # Build multimodal prompt
        qs = prompt
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        text_prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            text_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        # Preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        # Inference
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(self.device),
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=256,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"Output: {outputs}")
        return outputs

    def predict(self, image_input, prompt: str,
                conv_mode: str = "qwen_2",
                temperature: float = 0.2,
                top_p: float = None,
                num_beams: int = 1):
        """
        Inference function that accepts:
        - str : path to image file
        - PIL.Image.Image : already loaded image
        - torch.Tensor : preprocessed tensor from process_images()
        """
        from PIL import Image

        # --- Handle image input ---
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            logger.info(f"Inference on image path: {image_input}")
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
            # You can’t get width/height from tensor alone reliably; assume model input size
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

        # --- Inference ---
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(self.device),
                image_sizes=[image_size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=256,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"Output: {outputs}")
        return outputs


# -----------------------------------------------------
# FLASK APP
# -----------------------------------------------------
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)
vlm_server = FastVLMServer(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    start_total = time.perf_counter()
    try:
        data = request.get_json(force=True)
        image_path = data.get("image_path")
        prompt = data.get("prompt", "Describe the image.")

        if not image_path:
            return jsonify({"error": "Missing 'image_path'"}), 400

        logger.info(f"🖼️  Received prediction request | image: {image_path}, prompt: '{prompt[:60]}...'")

        # Submit to thread pool, time model inference
        start_infer = time.perf_counter()
        future = executor.submit(vlm_server.predict, image_path, prompt)
        result = future.result()
        infer_time = time.perf_counter() - start_infer

        total_time = time.perf_counter() - start_total

        logger.info(
            f"✅ Inference complete | model_time={infer_time:.2f}s | total_time={total_time:.2f}s | "
            f"image='{os.path.basename(image_path)}'"
        )

        return jsonify({
            "output": result,
            "timing": {
                "model_time_sec": round(infer_time, 3),
                "total_time_sec": round(total_time, 3)
            }
        })

    except Exception as e:
        total_time = time.perf_counter() - start_total
        logger.exception(f"❌ Error during prediction (elapsed {total_time:.2f}s)")
        return jsonify({"error": str(e), "elapsed_sec": round(total_time, 3)}), 500

import cv2
import time
import os
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path, threshold=30.0):
    """Detect shot boundaries using PySceneDetect."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]


def extract_scene_keyframes(video_path, scenes, output_dir="/tmp/fastvlm_frames"):
    """Extract mid-frame from each scene."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_paths = []

    for i, (start, end) in enumerate(scenes):
        mid_time = (start + end) / 2.0
        frame_idx = int(mid_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        path = os.path.join(output_dir, f"scene_{i:03d}.jpg")
        cv2.imwrite(path, frame)
        frame_paths.append(path)

    cap.release()
    return frame_paths


def describe_video_frames(frame_paths):
    """Run FastVLM captioning with chained context."""
    context_text = ""
    captions = []

    for i, frame_path in enumerate(frame_paths):
        if context_text:
            prompt = (
                f"Previously: '{context_text}'. "
                f"Now describe what happens in this new scene."
            )
        else:
            prompt = (
                "Describe what is happening in this first scene of the video."
            )

        caption = vlm_server.predict(frame_path, prompt)
        captions.append(caption)
        context_text = caption
        logger.info(f"[Scene {i}] {caption}")

    return captions


@app.route("/summarize_video", methods=["POST"])
def summarize_video_endpoint():
    """Full video summarization pipeline."""
    start_total = time.perf_counter()
    try:
        data = request.get_json(force=True)
        video_path = data.get("video_path")
        threshold = float(data.get("scene_threshold", 30.0))

        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Missing or invalid 'video_path'"}), 400

        logger.info(f"🎬 Summarizing video: {video_path}")
        logger.info(f"Scene threshold: {threshold}")

        # Detect scenes
        t0 = time.perf_counter()
        scenes = detect_scenes(video_path, threshold)
        scene_time = time.perf_counter() - t0
        logger.info(f"Detected {len(scenes)} scenes in {scene_time:.2f}s")

        # Extract keyframes
        t1 = time.perf_counter()
        frames = extract_scene_keyframes(video_path, scenes)
        extract_time = time.perf_counter() - t1
        logger.info(f"Extracted {len(frames)} frames in {extract_time:.2f}s")

        # Generate captions
        t2 = time.perf_counter()
        captions = describe_video_frames(frames)
        caption_time = time.perf_counter() - t2
        logger.info(f"Generated {len(captions)} captions in {caption_time:.2f}s")

        # Combine into summary
        summary = " ".join(captions)

        total_time = time.perf_counter() - start_total
        logger.info(
            f"✅ Video summary complete | total_time={total_time:.2f}s | "
            f"scenes={len(scenes)} | frames={len(frames)}"
        )

        return jsonify({
            "summary": summary,
            "scenes_detected": len(scenes),
            "frames_used": len(frames),
            "captions": captions,
            "timing": {
                "scene_detection_sec": round(scene_time, 2),
                "frame_extraction_sec": round(extract_time, 2),
                "captioning_sec": round(caption_time, 2),
                "total_sec": round(total_time, 2)
            }
        })

    except Exception as e:
        total_time = time.perf_counter() - start_total
        logger.exception(f"❌ Error summarizing video (elapsed {total_time:.2f}s)")
        return jsonify({"error": str(e), "elapsed_sec": round(total_time, 2)}), 500


if __name__ == "__main__":
    logger.info("Starting FastVLM server on port 7860 ...")
    app.run(host="0.0.0.0", port=7860)




# curl -X POST http://localhost:7860/predict \
#      -H "Content-Type: application/json" \
#      -d '{"image_path": "/home/shang/dev/test_assets/styles/combined.png", "prompt": "What is happening in this image?"}'

# curl -X POST http://127.0.0.1:7860/summarize_video \
#      -H "Content-Type: application/json" \
#      -d '{
#            "video_path": "/home/shang/dev/test_assets/DQKRRwvDAY0__asset_1.mp4",
#            "scene_threshold": 25.0
#          }'
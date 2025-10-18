#!/usr/bin/env python3
"""
FastVLM Hybrid-Batch Gradio UI
- Supports multiple images per run (one shared prompt or multiple)
- Encodes all images in parallel; decodes one-by-one safely
- Includes ONNX/TensorRT export utilities
"""

import os, sys, time, traceback
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import gradio as gr
from PIL import Image

# --- repo imports (LLaVA/FastVLM) ---
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# ---------- Utility ----------
def get_model_name_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path))

# ---------- Torch Setup ----------
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE.upper()}")

# ---------- Load Model ----------
MODEL_PATH = "./checkpoints/llava-fastvithd_0.5b_stage3"
MODEL_BASE = None

print("Loading model (this may take a minute)...")
disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL_PATH, MODEL_BASE, model_name, device=DEVICE)
model = model.half().to(DEVICE) if DEVICE == "cuda" else model.to(DEVICE)
model.eval()
print("Model loaded and set to eval mode.")

# Optional xformers
try:
    model.enable_xformers_memory_efficient_attention()
    print("xFormers enabled.")
except Exception:
    pass

# ---------- Warmup ----------
def _warmup_model():
    if DEVICE != "cuda":
        print("Warm-up skipped (non-CUDA).")
        return
    try:
        dummy_img = torch.randn(1, 3, 384, 384, device=DEVICE, dtype=torch.float16)
        dummy_ids = torch.randint(0, 1000, (1, 8), device=DEVICE)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            _ = model.generate(dummy_ids, images=dummy_img, max_new_tokens=1)
        torch.cuda.synchronize()
        print("CUDA warm-up complete.")
    except Exception as e:
        print("Warm-up failed:", e)

_warmup_model()

# ---------- Resize Helper ----------
def resize_for_vlm(image: Image.Image, max_side: int = 480) -> Image.Image:
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return image
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return image.resize((new_w, new_h), Image.BICUBIC)


import numpy as np
from PIL import Image as PILImage
import io

def _to_pil(img_obj):
    """
    Convert a variety of inputs to PIL.Image (or raise).
    Accepts:
      - PIL.Image.Image
      - numpy arrays (H,W) or (H,W,3) uint8/float
      - bytes (jpeg/png data)
      - (filename, pil_image) tuples from gr.Gallery
    """
    # Tuple (filename, pil_image)
    if isinstance(img_obj, tuple) and len(img_obj) >= 2:
        img_obj = img_obj[1]

    # Already PIL
    if isinstance(img_obj, PILImage.Image):
        return img_obj

    # Numpy array
    if isinstance(img_obj, np.ndarray):
        # if float convert to 0-255
        arr = img_obj
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.dtype in (np.float32, np.float64) else arr.astype(np.uint8)
        # if grayscale expand to RGB
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 4:
            # RGBA -> RGB
            arr = arr[..., :3]
        return PILImage.fromarray(arr)

    # Bytes / memoryview
    if isinstance(img_obj, (bytes, bytearray, memoryview)):
        try:
            return PILImage.open(io.BytesIO(bytes(img_obj))).convert("RGB")
        except Exception:
            raise ValueError("Cannot parse image bytes")

    # Strings: maybe a base64 data URL or path
    if isinstance(img_obj, str):
        # try file path
        if os.path.exists(img_obj):
            return PILImage.open(img_obj).convert("RGB")
        # try base64 data URI
        try:
            if img_obj.startswith("data:"):
                import base64
                header, b64 = img_obj.split(",", 1)
                data = base64.b64decode(b64)
                return PILImage.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            pass
        raise ValueError("String input was not a path or data URI")

    raise ValueError(f"Unsupported image object type: {type(img_obj)}")


def describe_images_hybrid(images, prompt_input, temperature, num_beams, resize_option):
    """
    Robust hybrid-batch inference:
     - sanitize inputs (filter None, tuples, arrays)
     - encode images in parallel, decode sequentially
    """
    # Normalize 'images' that Gradio can pass as single image or list
    if images is None:
        return "No images provided."

    # Some Gradio widgets may give a single image (not a list). Normalize to list
    if not isinstance(images, (list, tuple)):
        images = [images]

    # Convert each entry to PIL safely
    valid_images = []
    skipped = []
    for idx, itm in enumerate(images):
        if itm is None:
            skipped.append((idx, "None"))
            continue
        try:
            pil = _to_pil(itm)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            valid_images.append(pil)
        except Exception as e:
            skipped.append((idx, str(e)))
            continue

    if len(valid_images) == 0:
        msg = f"No valid images after sanitization. Skipped {len(skipped)} item(s): {skipped[:5]}"
        print("[FastVLM] " + msg)
        return msg

    # --- handle prompts: allow one or multi-line per image ---
    if isinstance(prompt_input, str):
        prompts = [p.strip() for p in prompt_input.split("\n") if p.strip()]
        if len(prompts) == 0:
            return "Please supply at least one prompt."
        if len(prompts) == 1:
            prompts = prompts * len(valid_images)
        elif len(prompts) != len(valid_images):
            return f"Number of prompts ({len(prompts)}) must match number of valid images ({len(valid_images)})."
    else:
        prompts = [str(prompt_input)] * len(valid_images)

    # resize map
    resize_map = {"360p (fastest)": 360, "480p (balanced)": 480, "720p (high detail)": 720}
    max_side = resize_map.get(resize_option, 480)

    t_start = time.time()
    print(f"[FastVLM] Processing {len(valid_images)} valid image(s) on {DEVICE}...")

    # Step 1: preprocess all valid images
    processed_imgs = [resize_for_vlm(img, max_side=max_side) for img in valid_images]
    image_sizes = [img.size for img in processed_imgs]

    # Step 2: process_images (repo util) returns list of tensors
    try:
        image_tensors = process_images(processed_imgs, image_processor, model.config)
    except Exception as e:
        tb = traceback.format_exc()
        print("[FastVLM] process_images failed:", e, tb)
        return f"process_images failed: {e}"

    image_tensors = torch.stack(image_tensors, dim=0)
    if DEVICE == "cuda":
        image_tensors = image_tensors.to(device=DEVICE, dtype=torch.float16, non_blocking=True)
    else:
        image_tensors = image_tensors.to(device=DEVICE)

    # Step 3: encode images in parallel if supported
    try:
        with torch.inference_mode(), (torch.amp.autocast(device_type="cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.inference_mode()):
            if hasattr(model, "encode_images"):
                vision_embeds = model.encode_images(image_tensors)
                print(f"[FastVLM] Encoded {vision_embeds.shape[0]} images (vision_embeds shape: {vision_embeds.shape})")
            else:
                vision_embeds = image_tensors
                print("[FastVLM] encode_images not available; using raw image tensors")
    except Exception as e:
        print("[FastVLM] vision encoding failed:", e, traceback.format_exc())
        vision_embeds = image_tensors

    # Step 4: sequential generation (safe)
    results = []
    gen_times = []
    for i, (img_small, prompt) in enumerate(zip(processed_imgs, prompts)):
        # build prompt with image token
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(device=DEVICE, dtype=torch.long)

        gen_t0 = time.time()
        try:
            with torch.inference_mode(), (torch.amp.autocast(device_type="cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.inference_mode()):
                out = model.generate(
                    input_ids,
                    images=image_tensors[i:i+1],
                    image_sizes=[image_sizes[i]],
                    do_sample=False if temperature <= 0.0 else True,
                    temperature=float(temperature),
                    num_beams=int(num_beams),
                    max_new_tokens=128,
                    use_cache=True,
                )
        except Exception as e:
            print(f"[FastVLM] generation failed for image {i}: {e}", traceback.format_exc())
            results.append(f"🖼️ Image {i+1}: generation error: {e}")
            gen_times.append(0.0)
            continue

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        gen_t1 = time.time()
        gen_times.append(gen_t1 - gen_t0)

        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        clean = text.split("Question:")[0].split("\n\n")[0].strip()
        results.append(f"🖼️ Image {i+1}: {clean}")

    total_t = time.time() - t_start
    avg_gen = (sum(gen_times) / len([t for t in gen_times if t > 0]) ) if any(gen_times) else 0.0
    summary = f"\n\n⏱️ Total {total_t:.2f}s | Avg gen {avg_gen:.2f}s | {DEVICE.upper()} | {max_side}px"
    print(summary)
    return "\n\n".join(results) + summary



# ---------- Hybrid Batch Inference ----------
def describe_images_hybrid_old(images, prompt_input, temperature, num_beams, resize_option):
    if not images or len(images) == 0:
        return "Please upload one or more images."

    # handle single or multi prompts
    if isinstance(prompt_input, str):
        prompts = [p.strip() for p in prompt_input.split("\n") if p.strip()]
        if len(prompts) == 1:
            prompts = prompts * len(images)
        elif len(prompts) != len(images):
            return f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})."
    else:
        prompts = [str(prompt_input)] * len(images)

    resize_map = {"360p (fastest)": 360, "480p (balanced)": 480, "720p (high detail)": 720}
    max_side = resize_map.get(resize_option, 480)

    t_start = time.time()
    print(f"[FastVLM] Processing {len(images)} image(s) on {DEVICE}...")

    # Step 1: preprocess all images
    processed_imgs = [resize_for_vlm(img.convert("RGB"), max_side=max_side) for img in images]
    image_sizes = [img.size for img in processed_imgs]
    image_tensors = process_images(processed_imgs, image_processor, model.config)
    image_tensors = torch.stack(image_tensors, dim=0)
    if DEVICE == "cuda":
        image_tensors = image_tensors.to(device=DEVICE, dtype=torch.float16, non_blocking=True)
    else:
        image_tensors = image_tensors.to(device=DEVICE)

    # Step 2: Encode vision features (parallel)
    print("[FastVLM] Encoding vision features...")
    try:
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.inference_mode():
            vision_embeds = model.encode_images(image_tensors)
        print(f"[FastVLM] Encoded {len(images)} images.")
    except Exception:
        vision_embeds = image_tensors
        print("⚠️ Using direct tensors (encode_images not found).")

    # Step 3: Generate per image
    results = []
    gen_times = []
    for i, (img, prompt) in enumerate(zip(processed_imgs, prompts)):
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(device=DEVICE, dtype=torch.long)

        gen_t0 = time.time()
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.inference_mode():
            out = model.generate(
                input_ids,
                images=image_tensors[i:i+1],
                image_sizes=[image_sizes[i]],
                do_sample=False if temperature <= 0.0 else True,
                temperature=float(temperature),
                num_beams=int(num_beams),
                max_new_tokens=128,
                use_cache=True,
            )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        gen_t1 = time.time()
        gen_times.append(gen_t1 - gen_t0)

        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        clean = text.split("Question:")[0].split("\n\n")[0].strip()
        results.append(f"🖼️ Image {i+1}: {clean}")

    total_t = time.time() - t_start
    summary = f"\n\n⏱️ Total {total_t:.2f}s | Avg gen {sum(gen_times)/len(gen_times):.2f}s | {DEVICE.upper()} | {max_side}px"
    print(summary)
    return "\n\n".join(results) + summary

# ---------- ONNX Export ----------
def export_lm_to_onnx():
    out_msgs = []
    try:
        lm = getattr(model, "model", getattr(model, "language_model", model))
        lm.eval().to(DEVICE)
        if DEVICE == "cuda":
            lm.half()

        example_ids = torch.randint(0, 1000, (1, 32), dtype=torch.long, device=DEVICE)
        onnx_path = "fastvlm_lm.onnx"
        torch.onnx.export(
            lm,
            (example_ids,),
            onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
        )
        out_msgs.append(f"✅ Exported {onnx_path}")
    except Exception as e:
        out_msgs.append(traceback.format_exc())
    return "\n".join(out_msgs)

# ---------- TensorRT Build ----------
def build_tensorrt_engine():
    if not torch.cuda.is_available():
        return "❌ CUDA not available."
    os.system("pip install -q tensorrt polygraphy onnx-graphsurgeon")
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open("fastvlm_lm.onnx", "rb") as f:
        if not parser.parse(f.read()):
            return "❌ Failed to parse ONNX."
    t0 = time.time()
    engine = builder.build_engine(network, config)
    dur = time.time() - t0
    with open("fastvlm_lm.plan", "wb") as f:
        f.write(engine.serialize())
    return f"✅ TensorRT engine built (FP16) in {dur:.1f}s."

# ---------- Gradio UI ----------
with gr.Blocks(css=".bigbox textarea {min-height: 400px !important;}") as demo:
    gr.Markdown("## 🧠 FastVLM Hybrid-Batch (Gradio) — PyTorch + ONNX/TensorRT Utilities")

    with gr.Row():
        with gr.Column(scale=3):
            image_in = gr.Gallery(label="Upload one or more images", type="pil", show_label=True)
            prompt_in = gr.Textbox(lines=3, placeholder="Enter one prompt or multiple (newline separated)", label="Prompt(s)")
            temp_in = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
            beams_in = gr.Slider(1, 5, value=1, step=1, label="Num Beams")
            resize_in = gr.Dropdown(["360p (fastest)", "480p (balanced)", "720p (high detail)"], value="480p (balanced)", label="Resize")
            run_btn = gr.Button("🚀 Run Inference (Hybrid-Batch)")
            out_text = gr.Textbox(label="Output", lines=12, elem_classes=["bigbox"], interactive=True)

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Export / Build Utilities")
            export_btn = gr.Button("Export LM → ONNX")
            export_out = gr.Textbox(label="ONNX Export Log", lines=8)
            build_btn = gr.Button("Build TensorRT Engine (FP16)")
            build_out = gr.Textbox(label="TensorRT Build Log", lines=6)

    run_btn.click(fn=describe_images_hybrid, inputs=[image_in, prompt_in, temp_in, beams_in, resize_in], outputs=out_text)
    export_btn.click(fn=export_lm_to_onnx, outputs=export_out)
    build_btn.click(fn=build_tensorrt_engine, outputs=build_out)

demo.queue()
demo.launch(share=True)

#!/usr/bin/env python3
"""
fastvlm_ui_full.py

- Runs Gradio multimodal UI using the repo's FastVLM (LLaVA-based) model via PyTorch (works end-to-end).
- Provides buttons to export LM backbone to ONNX and build a TensorRT engine (FP16).
- Designed for Colab (T4) — does not force ONNX/TensorRT actions on startup.
"""

import os, sys, time, traceback
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import gradio as gr
from PIL import Image

# --- repo imports (llava / fastvlm) ---
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# ----- small helper that some forks don't expose -----
import os as _os
def get_model_name_from_path(path: str) -> str:
    return _os.path.basename(_os.path.normpath(path))

# ---------- Torch global settings ----------
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------- Device ----------
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE.upper()}")

# ---------- Model loading (change MODEL_PATH to your checkpoint) ----------
MODEL_PATH = "./checkpoints/llava-fastvithd_0.5b_stage3"  # change if different
MODEL_BASE = None

print("Loading model (this may take a minute)...")
disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL_PATH, MODEL_BASE, model_name, device=DEVICE)

# Move to device and fp16 if cuda
if DEVICE == "cuda":
    model = model.half().to(DEVICE)
else:
    model = model.to(DEVICE)
model.eval()
print("Model loaded and set to eval mode.")

# Try enabling xformers if present (silently)
try:
    model.enable_xformers_memory_efficient_attention()
    print("xFormers enabled.")
except Exception:
    pass

# Warm up to reduce first-call overhead
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

# ---------- Utilities: resizing ----------
def resize_for_vlm(image: Image.Image, max_side: int = 480) -> Image.Image:
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return image
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return image.resize((new_w, new_h), Image.BICUBIC)

# ---------- Inference function (Gradio) ----------
def describe_image(image: Image.Image, prompt: str, temperature: float, num_beams: int, resize_option: str):
    if image is None:
        return "Please upload an image."
    if not prompt.strip():
        return "Please enter a prompt."

    resize_map = {"360p (fastest)": 360, "480p (balanced)": 480, "720p (high detail)": 720}
    max_side = resize_map.get(resize_option, 480)

    t_start = time.time()

    # resize
    img_small = resize_for_vlm(image.convert("RGB"), max_side=max_side)

    # build prompt with image tokens (follow repo pattern)
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    final_prompt = conv.get_prompt()

    # tokenize prompt (this returns a token sequence with image token already)
    input_ids = tokenizer_image_token(final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(device=DEVICE, dtype=torch.long, non_blocking=True)

    # preprocess image via repo helper
    image_tensor = process_images([img_small], image_processor, model.config)[0].unsqueeze(0)
    if DEVICE == "cuda":
        image_tensor = image_tensor.to(device=DEVICE, dtype=torch.float16, non_blocking=True)
    else:
        image_tensor = image_tensor.to(device=DEVICE)

    gen_t0 = time.time()
    # tune some backend options
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # generation (PyTorch backend - full multimodal generate)
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.inference_mode():
        out = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[img_small.size],
            do_sample=False if temperature <= 0.0 else True,
            temperature=float(temperature),
            num_beams=int(num_beams),
            max_new_tokens=128,
            use_cache=True,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    gen_t1 = time.time()

    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    # cleanup/truncate for UI
    if "Question:" in text:
        text = text.split("Question:")[0].strip()
    if "\n\n" in text:
        text = text.split("\n\n")[0].strip()
    words = text.split()
    if len(words) > 60:
        text = " ".join(words[:60]).rstrip(" ,.;:") + "."

    elapsed = time.time() - t_start
    dbg = f"⏱️ Total {elapsed:.2f}s | gen {gen_t1-gen_t0:.2f}s on {DEVICE.upper()}"
    print(dbg)
    return f"{text}\n\n🕒 {elapsed:.2f}s · {max_side}px · {DEVICE.upper()}"

# ---------- Export LM (ONNX) helper ----------
def export_lm_to_onnx():
    """
    Export the inner language model backbone (lm) to ONNX.
    This exports only the pure transformer LM (no multimodal glue).
    """
    out_msgs = []
    try:
        # find the LM backbone inside model
        if hasattr(model, "model"):
            lm = model.model
            out_msgs.append("Using model.model as LM backbone.")
        elif hasattr(model, "language_model"):
            lm = model.language_model
            out_msgs.append("Using model.language_model as LM backbone.")
        else:
            lm = model
            out_msgs.append("Falling back to top-level model as LM backbone (may include extra logic).")

        lm.eval().to(DEVICE)
        if DEVICE == "cuda":
            lm.half()

        # example shapes (fixed for export)
        B, S = 1, 32
        example_ids = torch.randint(0, 1000, (B, S), dtype=torch.long, device=DEVICE)

        onnx_path = "fastvlm_lm.onnx"
        if os.path.exists(onnx_path):
            out_msgs.append(f"Existing ONNX found: {onnx_path} — overwriting.")

        out_msgs.append("Exporting LM to ONNX using legacy exporter (stable)...")
        # Use legacy exporter (works reliably for many LM-only modules)

        # find inner LM (Qwen2 backbone)
        try:
            lm = model.model
            print("✅ Using model.model (Qwen2) as export target.")
        except AttributeError:
            lm = model
            print("⚠️ model.model not found, falling back to full model.")

        # Disable use_cache in config to avoid duplicate arg issues
        if hasattr(lm, "config"):
            lm.config.use_cache = False
            lm.config.return_dict = False

        # example input (no use_cache)
        example_ids = torch.randint(0, 1000, (1, 32), dtype=torch.long, device=DEVICE)

        # Export only the LM
        torch.onnx.export(
            lm,                         # direct Qwen2 model, not wrapper
            (example_ids,),              # no keyword args
            "fastvlm_lm.onnx",
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
            verbose=False,
        )


        out_msgs.append(f"✅ ONNX export succeeded: {onnx_path}")

        # quick ONNXRuntime check if available
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"])
            res = sess.run(None, {"input_ids": example_ids.cpu().numpy()})
            out_msgs.append(f"✅ ONNXRuntime test OK — output shape {res[0].shape}")
        except Exception as e:
            out_msgs.append(f"⚠️ ONNXRuntime test failed or not available: {e}")

    except Exception as e:
        tb = traceback.format_exc()
        out_msgs.append("❌ ONNX export failed:")
        out_msgs.append(tb)
    return "\n".join(out_msgs)

# ---------- Build TensorRT engine from ONNX ----------
def build_tensorrt_engine():
    """
    Build a TensorRT engine (FP16) from fastvlm_lm.onnx.
    Returns status messages. Requires CUDA.
    """
    msgs = []
    if not torch.cuda.is_available():
        return "❌ CUDA not available on this runtime. TensorRT build skipped."

    onnx_path = "fastvlm_lm.onnx"
    if not os.path.exists(onnx_path):
        return "❌ ONNX file not found. Please run 'Export LM to ONNX' first."

    # install tensorrt bindings & helpers if missing
    msgs.append("Installing/ensuring TensorRT & helpers (may take a moment) ...")
    os.system("pip install -q tensorrt polygraphy onnx-graphsurgeon")

    try:
        import tensorrt as trt
    except Exception as e:
        msgs.append(f"❌ Failed to import tensorrt Python bindings: {e}")
        return "\n".join(msgs)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            msgs.append("❌ Failed to parse ONNX. Parser errors:")
            for i in range(parser.num_errors):
                msgs.append(str(parser.get_error(i)))
            return "\n".join(msgs)

    msgs.append("Building TensorRT engine (FP16). This can take several minutes ...")
    t0 = time.time()
    engine = builder.build_engine(network, config)
    dur = time.time() - t0
    if engine is None:
        msgs.append("❌ TensorRT build returned None engine.")
        return "\n".join(msgs)
    plan_path = "fastvlm_lm.plan"
    with open(plan_path, "wb") as f:
        f.write(engine.serialize())
    msgs.append(f"✅ TensorRT engine built: {plan_path} (took {dur:.1f}s).")
    return "\n".join(msgs)

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("## FastVLM (Gradio) — PyTorch + ONNX/TensorRT utilities")
    with gr.Row():
        with gr.Column(scale=3):
            image_in = gr.Image(type="pil", label="Upload Image")
            prompt_in = gr.Textbox(lines=2, placeholder="Describe the image...", label="Prompt")
            temp_in = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
            beams_in = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Num Beams")
            resize_in = gr.Dropdown(choices=["360p (fastest)", "480p (balanced)", "720p (high detail)"], value="480p (balanced)", label="Resize Mode")
            run_btn = gr.Button("Run Inference (PyTorch)")
            out_text = gr.Textbox(label="Output")
        with gr.Column(scale=1):
            gr.Markdown("### Export / Build Utilities")
            export_btn = gr.Button("Export LM → ONNX")
            export_out = gr.Textbox(label="ONNX Export Log", lines=8)
            build_btn = gr.Button("Build TensorRT Engine (FP16)")
            build_out = gr.Textbox(label="TensorRT Build Log", lines=6)
            # quick info
            gr.Markdown("Notes:\n- ONNX/TensorRT exports the LM backbone only (no full multimodal generate exported).\n- Use export first, then build engine. Building may take minutes.")

    run_btn.click(fn=describe_image, inputs=[image_in, prompt_in, temp_in, beams_in, resize_in], outputs=out_text)
    export_btn.click(fn=export_lm_to_onnx, inputs=[], outputs=export_out)
    build_btn.click(fn=build_tensorrt_engine, inputs=[], outputs=build_out)

demo.queue()
demo.launch(share=True)

#!/usr/bin/env python3
"""
FastVLM Gradio UI — optimized for CUDA (T4/L4) and Colab.
Adds optional BetterTransformer + torch.compile acceleration.
"""

import os, time, torch, gradio as gr
from PIL import Image
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# -------------------- GLOBAL TORCH SETTINGS --------------------
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# -------------------- DEVICE SETUP --------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"🚀 Using device: {DEVICE.upper()}")

# -------------------- MODEL LOADING --------------------
MODEL_PATH = "./checkpoints/llava-fastvithd_0.5b_stage3"
MODEL_BASE = None
disable_torch_init()
print(f"🔧 Loading FastVLM model from: {MODEL_PATH}")

model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, MODEL_BASE, model_name, device=DEVICE
)

# -------------------- MODEL PREP --------------------
if DEVICE == "cuda":
    model = model.half().to(DEVICE)
else:
    model = model.to(DEVICE)
model.eval()

# try xFormers if available
try:
    model.enable_xformers_memory_efficient_attention()
    print("✅ xFormers attention enabled.")
except Exception:
    print("⚠️ xFormers not available or incompatible.")

# -------------------- OPTIONAL: BetterTransformer --------------------
try:
    from optimum.bettertransformer import BetterTransformer
    model = BetterTransformer.transform(model)
    print("✅ BetterTransformer enabled.")
except ImportError:
    print("ℹ️ Installing Optimum for BetterTransformer ...")
    os.system("pip install -q optimum")
    try:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
        print("✅ BetterTransformer enabled after install.")
    except Exception as e:
        print("⚠️ BetterTransformer not applied:", e)
except Exception as e:
    print("⚠️ BetterTransformer failed:", e)

# -------------------- OPTIONAL: Torch.compile --------------------
try:
    model = torch.compile(model)
    print("✅ Model compiled with torch.compile.")
except Exception as e:
    print("⚠️ torch.compile failed:", e)

print("✅ Model ready for inference.")

# -------------------- WARM-UP --------------------
def _warmup():
    if DEVICE != "cuda":
        return
    try:
        dummy_image = torch.randn(1, 3, 384, 384, dtype=torch.float16, device=DEVICE)
        dummy_ids = torch.randint(0, 1000, (1, 8), device=DEVICE)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            _ = model.generate(dummy_ids, images=dummy_image, max_new_tokens=1)
        torch.cuda.synchronize()
        print("⚙️ CUDA warm-up complete.")
    except Exception as e:
        print("⚠️ Warm-up failed:", e)

_warmup()

# -------------------- IMAGE RESIZE --------------------
def resize_for_vlm(image: Image.Image, max_side: int = 480) -> Image.Image:
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return image
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    print(f"🖼️ Resizing image from {w}x{h} -> {new_w}x{new_h}")
    return image.resize((new_w, new_h), Image.BICUBIC)

# -------------------- INFERENCE --------------------
def describe_image(image: Image.Image, prompt: str, temperature: float, num_beams: int, resize_option: str):
    if image is None:
        return "Please upload an image."
    if not prompt.strip():
        return "Please enter a prompt."

    try:
        resize_map = {"360p (fastest)": 360, "480p (balanced)": 480, "720p (high detail)": 720}
        max_side = resize_map.get(resize_option, 480)

        t_start = time.time()
        t0 = time.time(); image = resize_for_vlm(image.convert("RGB"), max_side); t1 = time.time()

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()

        tok0 = time.time()
        input_ids = tokenizer_image_token(final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(device=DEVICE, dtype=torch.long, non_blocking=True)
        tok1 = time.time()

        proc0 = time.time()
        image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0)
        image_tensor = image_tensor.to(device=DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32, non_blocking=True)
        proc1 = time.time()

        gen0 = time.time()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                num_beams=1,
                max_new_tokens=128,
                use_cache=True,
            )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        gen1 = time.time()

        dec0 = time.time()
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        dec1 = time.time()

        if "Question:" in output:
            output = output.split("Question:")[0].strip()
        if "\n\n" in output:
            output = output.split("\n\n")[0].strip()
        words = output.split()
        if len(words) > 50:
            output = " ".join(words[:50]).rstrip(" ,.;:") + "."

        elapsed = time.time() - t_start
        print(f"⏱️ Total {elapsed:.2f}s | resize {t1-t0:.2f}s | tokenize {tok1-tok0:.2f}s | "
              f"preproc {proc1-proc0:.2f}s | gen {gen1-gen0:.2f}s | decode {dec1-dec0:.2f}s on {DEVICE.upper()}")

        return f"{output}\n\n🕒 {elapsed:.2f}s · {resize_option} · {DEVICE.upper()}"

    except Exception as e:
        import traceback; traceback.print_exc()
        return f"⚠️ Error during inference: {e}"

# -------------------- GRADIO UI --------------------
demo = gr.Interface(
    fn=describe_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(lines=2, placeholder="Prompt (e.g. 'Describe the image.')"),
        gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Temperature"),
        gr.Slider(1, 5, value=1, step=1, label="Num Beams"),
        gr.Dropdown(
            ["360p (fastest)", "480p (balanced)", "720p (high detail)"],
            value="480p (balanced)",
            label="Resize Mode",
        ),
    ],
    outputs=gr.Textbox(label="FastVLM Output"),
    title="FastVLM Interactive Demo (Optimized)",
    description="Auto-optimized for FP16 and CUDA. "
                "Includes optional BetterTransformer and torch.compile for faster inference.",
)

demo.queue()
demo.launch(share=True)

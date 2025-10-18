#!/usr/bin/env python3
"""
FastVLM Gradio UI
Faithful to predict.py but optimized for MPS / CUDA.
Adds dynamic image resizing with dropdown presets.
"""

import gradio as gr
import torch
from PIL import Image
import time

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

# ---------- DEVICE SETUP ----------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"🚀 Using device: {DEVICE.upper()}")

# ---------- MODEL LOADING ----------
MODEL_PATH = "./checkpoints/llava-fastvithd_0.5b_stage3"
MODEL_BASE = None

print(f"🔧 Loading FastVLM model from: {MODEL_PATH}")
disable_torch_init()

model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, MODEL_BASE, model_name, device=DEVICE
)

print("✅ Model loaded successfully and ready for inference.")


# ---------- IMAGE RESIZING ----------
def resize_for_vlm(image: Image.Image, max_side: int = 480) -> Image.Image:
    """
    Resize image so the longest side = max_side, preserving aspect ratio.
    """
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return image  # already small enough
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    print(f"🖼️ Resizing image from {w}x{h} -> {new_w}x{new_h}")
    return image.resize((new_w, new_h), Image.BICUBIC)


# ---------- INFERENCE ----------
def describe_image(image: Image.Image, prompt: str, temperature: float, num_beams: int, resize_option: str):
    """Describe an image using FastVLM — same logic as predict.py."""
    if image is None:
        return "Please upload an image."
    if not prompt.strip():
        return "Please enter a prompt."

    try:
        # Map dropdown label to numeric max side
        resize_map = {"360p (fastest)": 360, "480p (balanced)": 480, "720p (high detail)": 720}
        max_side = resize_map.get(resize_option, 480)

        start_time = time.time()

        # --- Step 1: Resize for speed ---
        image = resize_for_vlm(image.convert("RGB"), max_side=max_side)

        # --- Step 2: Add image tokens to prompt ---
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        # --- Step 3: Conversation template ---
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()

        # --- Step 4: Tokenize prompt ---
        input_ids = (
            tokenizer_image_token(final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(DEVICE)
        )

        # --- Step 5: Preprocess image ---
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().to(DEVICE)

        # --- Step 6: Generate output ---
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=256,
                use_cache=True,
            )

        # --- Step 7: Decode ---
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Cleanup
        if "Question:" in output:
            output = output.split("Question:")[0].strip()
        if "\n\n" in output:
            output = output.split("\n\n")[0].strip()

        # Trim to ~50 words if necessary
        words = output.split()
        if len(words) > 50:
            output = " ".join(words[:50]).rstrip(" ,.;:") + "."

        elapsed = time.time() - start_time
        print(f"⏱️ Inference took {elapsed:.2f} seconds using {resize_option} on {DEVICE.upper()}")

        return f"{output}\n\n🕒 {elapsed:.2f}s · {resize_option} · {DEVICE.upper()}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ Error during inference: {e}"


# ---------- GRADIO UI ----------
demo = gr.Interface(
    fn=describe_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(lines=2, placeholder="Enter a prompt (e.g. 'Describe the image under 50 words.')"),
        gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Temperature"),
        gr.Slider(1, 5, value=1, step=1, label="Num Beams"),
        gr.Dropdown(
            choices=["360p (fastest)", "480p (balanced)", "720p (high detail)"],
            value="480p (balanced)",
            label="Resize Mode",
        ),
    ],
    outputs=gr.Textbox(label="FastVLM Output"),
    title="FastVLM Interactive Demo (Apple, 2025)",
    description="Faithful Gradio replica of predict.py. "
                "Choose resize preset to trade off speed and quality. "
                "Smaller images → faster inference.",
)

demo.launch(server_name="0.0.0.0", server_port=7860)

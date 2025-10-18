#!/usr/bin/env python3
"""
onnx_decoder.py
Greedy autoregressive decoder using ONNXRuntime.
Assumes:
 - fastvlm_lm.onnx exists (exports LM that accepts input_ids -> logits)
 - tokenizer from your repo is importable and matches model vocab
 - This decoder re-forwards the entire sequence each step (no KV cache)
"""

import time, os, sys, numpy as np
import torch

# change if needed:
ONNX_PATH = "fastvlm_lm.onnx"
MAX_NEW_TOKENS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(f"{ONNX_PATH} not found — run the ONNX export first.")

# import repo tokenizer (adjust import path if your repo layout differs)
from llava.mm_utils import tokenizer_image_token, process_images  # for tokenization helpers
from llava.model.builder import load_pretrained_model, get_model_name_from_path  # IF available
from llava.utils import disable_torch_init

# load tokenizer via the same path/codepath you used to export model (quick helper)
MODEL_PATH = "./checkpoints/llava-fastvithd_0.5b_stage3"
disable_torch_init()
# load only tokenizer & model metadata — this may be heavy but required to get tokenizer
tokenizer, model_dummy, image_processor, context_len = load_pretrained_model(MODEL_PATH, None, None, device="cpu")
# put model_dummy away
del model_dummy

# load onnxruntime
import onnxruntime as ort

# Create session with CUDA EP if possible
providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" or ("CUDAExecutionProvider" in ort.get_all_providers()) and DEVICE == "cuda" else ["CPUExecutionProvider"]
sess = ort.InferenceSession(ONNX_PATH, providers=providers)

# Print input / output metadata
print("ONNX inputs:", [i.name + str(i.shape) for i in sess.get_inputs()])
print("ONNX outputs:", [o.name + str(o.shape) for o in sess.get_outputs()])
input_name = sess.get_inputs()[0].name  # typically "input_ids"
output_name = sess.get_outputs()[0].name  # typically "logits" or "output"

# helper: run greedy decode
def onnx_greedy_decode(prompt_text, max_new_tokens=MAX_NEW_TOKENS):
    # Build tokenized input (single batch)
    # tokenization must produce same ids as used during export
    # For your repo you used tokenizer_image_token to inject image token, but LM-only expects input ids
    # So supply a text-only prompt here (LM-only). For multimodal prompts, you must construct tokens similarly.
    enc = tokenizer(prompt_text)
    input_ids = enc.input_ids if hasattr(enc, "input_ids") else enc["input_ids"]
    # ensure it's a list of ints
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    input_ids = list(input_ids)
    for _ in range(max_new_tokens):
        # prepare numpy input (batch 1)
        ids_np = np.array([input_ids], dtype=np.int64)  # ONNX expects int64 token ids in many cases
        # run ONNX
        t0 = time.time()
        ort_outs = sess.run([output_name], {input_name: ids_np})
        t1 = time.time()
        logits = ort_outs[0]  # shape (batch, seq, vocab) or (batch, vocab) depending on export
        # if logits shape is (batch, seq, vocab), pick last token's logits
        if logits.ndim == 3:
            last_logits = logits[0, -1, :]
        elif logits.ndim == 2:
            # shape (batch, vocab) — already last-token logits
            last_logits = logits[0]
        else:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        next_id = int(np.argmax(last_logits))
        input_ids.append(next_id)
        # small stopping criteria if eos present in tokenizer
        try:
            eos = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None
            if eos is not None and next_id == eos:
                break
        except Exception:
            pass
    return input_ids

# Quick benchmark: feed a short prompt and compare runtimes
PROMPT = "The city skyline at sunset shows"
print("Prompt:", PROMPT)
t0 = time.time()
ids_final = onnx_greedy_decode(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
t1 = time.time()
text_out = tokenizer.decode(ids_final, skip_special_tokens=True) if hasattr(tokenizer, "decode") else "decoded_text_unavailable"
print("Decoded text (truncated):", text_out[:400])
print(f"ONNX greedy decode time for {MAX_NEW_TOKENS} tokens: {t1-t0:.3f}s (avg {(t1-t0)/MAX_NEW_TOKENS:.4f}s/token)")

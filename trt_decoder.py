#!/usr/bin/env python3

# !pip install -q pycuda
"""
trt_decoder.py
Greedy decoder using TensorRT engine file fastvlm_lm.plan
Note: requires tensorrt python bindings and pycuda.
This runs full-seq forward each step (no KV cache).
"""

import os, time, numpy as np
import torch

ENGINE_PATH = "fastvlm_lm.plan"
MAX_NEW_TOKENS = 32

if not os.path.exists(ENGINE_PATH):
    raise FileNotFoundError(f"{ENGINE_PATH} not found. Build it first.")

# repo tokenizer
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
MODEL_PATH = "./checkpoints/llava-fastvithd_0.5b_stage3"
disable_torch_init()
tokenizer, _, _, _ = load_pretrained_model(MODEL_PATH, None, None, device="cpu")

# Import tensorrt & pycuda
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# load engine
with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# find binding indices & shapes
n_bindings = engine.num_bindings
bindings = [None] * n_bindings
binding_names = [engine.get_binding_name(i) for i in range(n_bindings)]
print("Engine bindings:", binding_names)

# allocate device memory for inputs/outputs based on binding shapes (use max dims)
d_input_name = binding_names[0]
d_output_name = binding_names[-1]

# helper to allocate buffer
def allocate_binding(engine, name):
    idx = engine.get_binding_index(name)
    dtype = trt.nptype(engine.get_binding_dtype(idx))
    # use max shape (engine.max_batch_size or binding shape)
    shape = engine.get_binding_shape(idx)
    # if dynamic, use reasonable max (e.g., batch=1, seq=128)
    shape = [dim if dim > 0 else 1 for dim in shape]
    size = int(np.prod(shape))
    host_mem = np.empty(shape, dtype=dtype)
    dev_mem = cuda.mem_alloc(host_mem.nbytes)
    return idx, host_mem, dev_mem

# simplistic allocation for first input and first output
in_idx, in_host, in_dev = allocate_binding(engine, d_input_name)
out_idx, out_host, out_dev = allocate_binding(engine, d_output_name)

stream = cuda.Stream()

def trt_infer(input_ids_np):
    # copy to host buffer (resize if needed)
    in_host_flat = np.array(input_ids_np, dtype=in_host.dtype)
    # if host buffer size mismatches, reallocate (simpler to require shapes match export)
    cuda.memcpy_htod_async(in_dev, in_host_flat.tobytes(), stream)
    # run inference
    context.execute_async_v2(bindings=[int(in_dev), int(out_dev)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(out_host, out_dev, stream)
    stream.synchronize()
    # reshape output to (batch, seq, vocab) or (batch, vocab) depending
    return out_host.copy()

# greedy loop
def trt_greedy(prompt_text, max_new_tokens=MAX_NEW_TOKENS):
    enc = tokenizer(prompt_text)
    input_ids = enc.input_ids if hasattr(enc, "input_ids") else enc["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    input_ids = list(input_ids)
    for _ in range(max_new_tokens):
        ids_np = np.array([input_ids], dtype=in_host.dtype)
        out_logits = trt_infer(ids_np)
        # parse logits
        if out_logits.ndim == 3:
            last = out_logits[0, -1, :]
        else:
            last = out_logits[0]
        next_id = int(np.argmax(last))
        input_ids.append(next_id)
        try:
            eos = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None
            if eos is not None and next_id == eos:
                break
        except Exception:
            pass
    return input_ids

# benchmark
PROMPT = "The city skyline at sunset shows"
t0 = time.time()
ids_final = trt_greedy(PROMPT, MAX_NEW_TOKENS)
t1 = time.time()
print("Decoded (truncated):", tokenizer.decode(ids_final)[:400])
print(f"TensorRT greedy decode time: {t1-t0:.3f}s; avg {(t1-t0)/MAX_NEW_TOKENS:.4f}s/token")

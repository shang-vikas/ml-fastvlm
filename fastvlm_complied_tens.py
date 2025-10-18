#!/usr/bin/env python3
"""
FastVLM ONNX + TensorRT exporter (Colab-ready, PyTorch ≥ 2.3)
Exports only the transformer forward path (no generate() loop)
"""

import os, torch, traceback, time

# --- 0. install dependencies (quietly) ---
os.system("pip install -q onnx onnxscript onnxruntime-gpu tensorrt polygraphy onnx-graphsurgeon")

# --- 1. imports from your FastVLM repo ---
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model

# --- 2. envs for reproducible export ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_SDP_KERNEL"] = "math"     # use traceable attention
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# --- 3. load FastVLM model ---
MODEL_PATH = "./checkpoints/llava-fastvithd_0.5b_stage3"
disable_torch_init()

# fallback helper
def get_model_name_from_path(path: str) -> str:
    """Mimic original LLaVA util: return folder name or last checkpoint token."""
    return os.path.basename(os.path.normpath(path))

model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, None, model_name, device=DEVICE
)
model.eval().to(DEVICE)
if DEVICE == "cuda": model.half()
print("✅ Model loaded for export.")

# --- 4. wrapper for stable exportable forward ---
class ExportWrapper(torch.nn.Module):
    """Expose stable forward(input_ids, images) → tensor"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, images):
        out = self.model(input_ids=input_ids, images=images)
        if hasattr(out, "logits"):
            return out.logits
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, tuple) and torch.is_tensor(out[0]):
            return out[0]
        raise RuntimeError("No tensor output found in model forward()")

wrapper = ExportWrapper(model).to(DEVICE)
if DEVICE == "cuda": wrapper.half()
wrapper.eval()

# --- 5. example inputs for tracing ---
B, S, H, W = 1, 32, 384, 384
example_ids = torch.randint(0, 1000, (B, S), dtype=torch.long, device=DEVICE)
example_img = torch.randn(B, 3, H, W, dtype=torch.float16 if DEVICE=="cuda" else torch.float32, device=DEVICE)

# --- 6. export to ONNX with new exporter ---
onnx_file = "fastvlm_export.onnx"
try:
    print("🧱 Exporting ONNX with torch.onnx.export(..., dynamo=True) ...")
    torch.onnx.export(
        wrapper,
        (example_ids, example_img),
        onnx_file,
        input_names=["input_ids", "images"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=True,                     # use new exporter
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},  # dynamic batch/seq
        },
    )
    print(f"✅ ONNX export success → {onnx_file}")
except Exception:
    print("❌ ONNX export failed:")
    traceback.print_exc()
    raise SystemExit(1)

# --- 7. quick ONNXRuntime check ---
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_file, providers=["CUDAExecutionProvider"])
    out = sess.run(None, {
        "input_ids": example_ids.cpu().numpy(),
        "images": example_img.cpu().numpy(),
    })
    print(f"✅ ONNXRuntime inference ok. Output shape: {out[0].shape}")
except Exception:
    print("⚠️ ONNXRuntime check failed (you can still build TensorRT).")

# --- 8. build TensorRT engine (.plan) ---
engine_path = "fastvlm_export.plan"
if DEVICE != "cuda":
    print("⚠️ TensorRT skipped (CUDA not available).")
else:
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    parser = trt.OnnxParser(builder.create_network(network_flags), TRT_LOGGER)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ TensorRT ONNX parse errors:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(1)

    print("⚙️ Building TensorRT engine (FP16)... may take several minutes")
    t0 = time.time()
    network = parser.network
    engine = builder.build_engine(network, config)
    dur = time.time() - t0
    if engine:
        with open(engine_path, "wb") as f: f.write(engine.serialize())
        print(f"✅ TensorRT engine built → {engine_path} (took {dur:.1f}s)")
    else:
        print("❌ TensorRT engine build failed.")

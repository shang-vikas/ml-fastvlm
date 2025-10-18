import torch
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# FastVLM imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

# --- Configuration ---
MODEL_PATH = "./checkpoints/llava-fastvithd_0.5b_stage3"
MODEL_NAME = "llava-fastvithd_0.5b"  # required by FastVLM loader
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Initialize app ---
app = FastAPI(title="FastVLM API Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model once ---
print(f"🚀 Loading FastVLM model '{MODEL_NAME}' from {MODEL_PATH} onto {DEVICE} ...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=None,
    model_name=MODEL_NAME,
    device=DEVICE,
)
model.eval()
print("✅ Model loaded and ready for inference.")


# --- Helper function ---
def describe(image_bytes: bytes, prompt: str) -> str:
    """Run inference on a single image and text prompt."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config).to(DEVICE, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX=0, return_tensors="pt").unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=128,
            do_sample=False,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# --- API endpoint ---
@app.post("/describe")
async def describe_image(file: UploadFile, prompt: str = Form(...)):
    try:
        image_bytes = await file.read()
        result = describe(image_bytes, prompt)
        return JSONResponse({"description": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


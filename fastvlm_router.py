# fastvlm_router.py
import os
import time
import logging
import subprocess
from typing import List, Dict, Any
from itertools import cycle
import psutil


import requests
from flask import Flask, request, jsonify

try:
    import pynvml
except ImportError:
    pynvml = None

import threading  # NEW



logging.basicConfig(
    level=os.getenv("FASTVLM_LOG_LEVEL", "INFO"),
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FastVLMRouter")

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
GPU_INDEX = int(os.getenv("FASTVLM_GPU_INDEX", "0"))
BACKEND_BASE_PORT = int(os.getenv("FASTVLM_BACKEND_BASE_PORT", "7860"))
ROUTER_PORT = int(os.getenv("FASTVLM_ROUTER_PORT", "9000"))
TARGET_VRAM_FRACTION = float(os.getenv("FASTVLM_TARGET_VRAM_FRACTION", "0.4"))
MAX_WORKERS = int(os.getenv("FASTVLM_MAX_WORKERS", "3"))
PYTHON_BIN = os.getenv("FASTVLM_PYTHON_BIN", "python3")
SERVER_MODULE = os.getenv(
    "FASTVLM_SERVER_MODULE",
    "pugsy_ai.pipelines.vlm_pipeline.fastvlm.ml-fastvlm.fastvlm_server",
)
MAX_CONCURRENT_PER_WORKER = int(os.getenv("FASTVLM_MAX_CONCURRENT_PER_WORKER", "2"))


TARGET_VRAM_FRACTION = float(os.getenv("FASTVLM_TARGET_VRAM_FRACTION", "0.7"))
TARGET_RAM_FRACTION = float(os.getenv("FASTVLM_TARGET_RAM_FRACTION", "0.8"))  # NEW
MAX_WORKERS = int(os.getenv("FASTVLM_MAX_WORKERS", "4"))


CHECK_READY_TIMEOUT_SEC = 300
CHECK_READY_INTERVAL_SEC = 2


# -----------------------------
# GPU helpers
# -----------------------------
def init_nvml():
    if pynvml is None:
        raise RuntimeError("pynvml is not installed. pip install pynvml")
    pynvml.nvmlInit()
    logger.info("Initialized NVML")


def get_gpu_mem_usage(gpu_index: int) -> Dict[str, int]:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "total": meminfo.total,
        "used": meminfo.used,
        "free": meminfo.free,
    }


def get_ram_usage() -> Dict[str, float]:
    """Return system RAM usage in bytes and fraction used."""
    vm = psutil.virtual_memory()
    return {
        "total": float(vm.total),
        "used": float(vm.used),
        "free": float(vm.available),
        "used_frac": float(vm.used) / float(vm.total),
    }


# -----------------------------
# Worker management
# -----------------------------
class Worker:
    def __init__(self, port: int, process: subprocess.Popen):
        self.port = port
        self.process = process
        self.in_flight = 0  # NEW
        self.lock = threading.Lock()  # NEW

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def try_acquire_slot(self) -> bool:
        """
        Try to reserve one concurrency slot for this worker.
        Returns True if successful, False if worker is at max.
        """
        with self.lock:
            if self.in_flight >= MAX_CONCURRENT_PER_WORKER:
                return False
            self.in_flight += 1
            return True

    def release_slot(self) -> None:
        """
        Release a previously acquired concurrency slot.
        Safe to call in finally-block.
        """
        with self.lock:
            if self.in_flight > 0:
                self.in_flight -= 1

workers: List[Worker] = []
worker_cycle = None  # will be cycle([...]) once workers exist
workers_lock = threading.Lock()  # NEW



def spawn_worker(port: int) -> Worker:
    env = os.environ.copy()
    env["FASTVLM_PORT"] = str(port)

    logger.info("Spawning FastVLM worker on port %d ...", port)
    # Run: python -m <server_module>
    proc = subprocess.Popen(
        [
            PYTHON_BIN,
            "-m",
            SERVER_MODULE,
        ],
        env=env,
    )
    return Worker(port=port, process=proc)


def wait_until_ready(worker: Worker, timeout_sec: int = CHECK_READY_TIMEOUT_SEC) -> bool:
    deadline = time.time() + timeout_sec
    url = f"{worker.base_url}/readyz"
    logger.info("Waiting for worker on %s to become ready ...", url)

    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("Worker on %s is ready.", url)
                return True
        except Exception:
            pass
        time.sleep(CHECK_READY_INTERVAL_SEC)

    logger.error("Worker on %s did not become ready within %d seconds.", url, timeout_sec)
    return False


# def bootstrap_workers_legacy_gpu_only():
#     global workers, worker_cycle

#     init_nvml()

#     # Spawn at least one worker
#     next_port = BACKEND_BASE_PORT
#     for i in range(MAX_WORKERS):
#         mem = get_gpu_mem_usage(GPU_INDEX)
#         used_frac = mem["used"] / mem["total"]
#         logger.info(
#             "GPU %d memory usage before spawning worker #%d: used=%.1f%% (%.2f GiB / %.2f GiB)",
#             GPU_INDEX,
#             i + 1,
#             used_frac * 100,
#             mem["used"] / (1024**3),
#             mem["total"] / (1024**3),
#         )

#         if i > 0 and used_frac >= TARGET_VRAM_FRACTION:
#             logger.info(
#                 "Stopping worker spawn: used_frac=%.3f >= target=%.3f",
#                 used_frac,
#                 TARGET_VRAM_FRACTION,
#             )
#             break

#         w = spawn_worker(next_port)
#         if not wait_until_ready(w):
#             logger.error("Worker on port %d failed to become ready, terminating process.", next_port)
#             w.process.terminate()
#             break

#         workers.append(w)
#         next_port += 1

#     if not workers:
#         raise RuntimeError("No FastVLM workers could be started.")

#     worker_cycle = cycle(workers)
#     logger.info("Started %d FastVLM workers.", len(workers))

def bootstrap_workers():
    global workers, worker_cycle

    init_nvml()

    next_port = BACKEND_BASE_PORT
    for i in range(MAX_WORKERS):
        mem = get_gpu_mem_usage(GPU_INDEX)
        ram = get_ram_usage()

        used_frac = mem["used"] / mem["total"]
        ram_used_frac = ram["used_frac"]

        logger.info(
            "Before spawning worker #%d: GPU used=%.1f%% (%.2f GiB / %.2f GiB), "
            "RAM used=%.1f%% (%.2f GiB / %.2f GiB)",
            i + 1,
            used_frac * 100,
            mem["used"] / (1024**3),
            mem["total"] / (1024**3),
            ram_used_frac * 100,
            ram["used"] / (1024**3),
            ram["total"] / (1024**3),
        )

        if i > 0 and used_frac >= TARGET_VRAM_FRACTION:
            logger.info(
                "Stopping worker spawn due to GPU: used_frac=%.3f >= target=%.3f",
                used_frac,
                TARGET_VRAM_FRACTION,
            )
            break

        if i > 0 and ram_used_frac >= TARGET_RAM_FRACTION:
            logger.info(
                "Stopping worker spawn due to RAM: used_frac=%.3f >= target=%.3f",
                ram_used_frac,
                TARGET_RAM_FRACTION,
            )
            break

        w = spawn_worker(next_port)
        if not wait_until_ready(w):
            logger.error("Worker on port %d failed to become ready, terminating process.", next_port)
            w.process.terminate()
            break

        workers.append(w)
        next_port += 1

    if not workers:
        raise RuntimeError("No FastVLM workers could be started.")

    worker_cycle = cycle(workers)
    logger.info("Started %d FastVLM workers.", len(workers))


def pick_available_worker() -> Worker:
    """
    Pick a worker that has available concurrency.
    Tries each worker at most once; if none have capacity, raises.
    """
    global worker_cycle

    if worker_cycle is None or not workers:
        raise RuntimeError("Worker cycle not initialized or no workers available.")

    with workers_lock:
        # Snapshot to avoid infinite loop
        num_workers = len(workers)
        for _ in range(num_workers):
            w = next(worker_cycle)
            if w.try_acquire_slot():
                return w

    # If we reach here, no worker had capacity
    raise RuntimeError("no_worker_capacity")

## helper to compute retry_after_sec for 503 responses
def compute_retry_after(endpoint: str, retry_attempt: int, in_flight_hint: int = 0) -> float:
    """
    Compute adaptive retry_after_sec based on:
    - endpoint (/summarize_video vs /predict_image)
    - retry attempt index (0,1,2,...)
    - rough inflight load hint (number of active requests)
    """

    # Heavier endpoint: assume long jobs
    if endpoint == "/summarize_video":
        base = 8.0   # minimal wait
        per_load = in_flight_hint * 10.0  # +10s per inflight
        growth = min(60.0, 2.0 ** retry_attempt)
    else:
        # image or others: cheaper
        base = 1.0
        per_load = in_flight_hint * 2.0
        growth = min(10.0, 2.0 ** retry_attempt)

    return base + per_load + growth


# -----------------------------
# Proxy helpers
# -----------------------------
def proxy_get(path: str):
    try:
        w = pick_available_worker()
    except RuntimeError as e:
        if str(e) == "no_worker_capacity":
            retry_attempt = int(request.headers.get("X-Retry-Attempt", "0"))
            with workers_lock:
                total_in_flight = sum(wk.in_flight for wk in workers)
            retry_after = compute_retry_after(request.path, retry_attempt, in_flight_hint=total_in_flight)

            body = jsonify({
                "error": {
                    "code": "workers_busy",
                    "message": "All FastVLM workers are at max concurrency. Please retry after some time.",
                    "retry_after_sec": retry_after,
                }
            })
            return body, 503, [("Retry-After", str(retry_after))]
        # some other bug
        raise

    try:
        url = f"{w.base_url}{path}"
        r = requests.get(url, params=request.args, timeout=300)
        headers = [(k, v) for k, v in r.headers.items()]
        return (r.content, r.status_code, headers)
    finally:
        w.release_slot()

def proxy_post(path: str):
    try:
        w = pick_available_worker()
    except RuntimeError as e:
        if str(e) == "no_worker_capacity":
            retry_attempt = int(request.headers.get("X-Retry-Attempt", "0"))
            with workers_lock:
                total_in_flight = sum(wk.in_flight for wk in workers)
            retry_after = compute_retry_after(request.path, retry_attempt, in_flight_hint=total_in_flight)

            body = jsonify({
                "error": {
                    "code": "workers_busy",
                    "message": "All FastVLM workers are at max concurrency. Please retry after some time.",
                    "retry_after_sec": retry_after,
                }
            })
            return body, 503, [("Retry-After", str(retry_after))]
        # anything else is a real bug
        raise

    try:
        url = f"{w.base_url}{path}"
        json_body = request.get_json(force=True, silent=True)
        r = requests.post(url, json=json_body, timeout=300)
        headers = [(k, v) for k, v in r.headers.items()]
        return (r.content, r.status_code, headers)
    finally:
        w.release_slot()


# -----------------------------
# Router endpoints
# -----------------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    with workers_lock:
        info = [
            {
                "port": w.port,
                "in_flight": w.in_flight,
                "max_concurrent": MAX_CONCURRENT_PER_WORKER,
            }
            for w in workers
        ]
    return jsonify({"status": "ok", "workers": info}), 200


@app.route("/readyz", methods=["GET"])
def readyz():
    # Router is ready if at least one worker exists
    if not workers:
        return jsonify({"status": "no_workers"}), 503
    return jsonify({"status": "ready", "workers": len(workers)}), 200


@app.route("/predict_image", methods=["POST"])
def predict_image():
    return proxy_post("/predict_image")


@app.route("/summarize_video", methods=["POST"])
def summarize_video():
    return proxy_post("/summarize_video")


if __name__ == "__main__":
    bootstrap_workers()
    logger.info("Starting FastVLM Router on 0.0.0.0:%d", ROUTER_PORT)
    app.run(host="0.0.0.0", port=ROUTER_PORT, debug=False)



# this is how client should backoff when receiving 503 workers_busy
# resp = requests.post(url, json=payload)
# if resp.status_code == 503:
#     data = resp.json()
#     if data.get("error", {}).get("code") == "workers_busy":
#         time.sleep(data["error"].get("retry_after_sec", 5))
#         # then retry or bubble this up with a clear message 

"""Manage remote whisper and vLLM servers on 5090 via SSH."""
import subprocess
import time
import httpx


def ssh_run(cmd: str, timeout: int = 10) -> str:
    r = subprocess.run(
        ["ssh", "wsl", cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return r.stdout.strip()


def ensure_ssh_tunnel(local_port: int) -> bool:
    """Ensure SSH tunnel exists for given port. Return True if created/exists."""
    r = subprocess.run(["lsof", f"-ti:{local_port}"], capture_output=True, text=True)
    if r.stdout.strip():
        return True
    result = subprocess.run(
        ["ssh", "-f", "-N", "-L", f"{local_port}:localhost:{local_port}", "wsl"],
        capture_output=True, timeout=10
    )
    return result.returncode == 0


def check_whisper_server() -> bool:
    try:
        r = httpx.get("http://localhost:8787/docs", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def start_whisper_server():
    print("[server] Starting whisper server on 5090...", flush=True)
    ssh_run(
        "nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && "
        "conda activate medllm && python3 /tmp/whisper_server.py' "
        "> /tmp/whisper_server.log 2>&1 &",
        timeout=5
    )
    for i in range(30):
        time.sleep(2)
        if check_whisper_server():
            print("[server] Whisper server ready.", flush=True)
            return True
        if i % 5 == 4:
            print(f"[server] Waiting for whisper... ({(i+1)*2}s)", flush=True)
    print("[server] Whisper server failed to start!", flush=True)
    return False


def check_vllm_server(expected_model: str = None) -> tuple[bool, str | None]:
    """Check if vLLM is running and serving the expected model.
    Returns (is_running, current_model_or_None).
    """
    try:
        r = httpx.get("http://localhost:8000/v1/models", timeout=3.0)
        if r.status_code != 200:
            return False, None
        models = r.json().get("data", [])
        if not models:
            return True, None
        current = models[0].get("id", "")
        if expected_model and current != expected_model:
            return True, current  # wrong model
        return True, current
    except Exception:
        return False, None


def stop_vllm_server():
    print("[server] Stopping vLLM...", flush=True)
    ssh_run("pkill -f 'vllm serve' 2>/dev/null; sleep 2")
    # Wait for port to free
    for _ in range(10):
        running, _ = check_vllm_server()
        if not running:
            return
        time.sleep(1)


def start_vllm_server(model: str, quantize: bool = True):
    print(f"[server] Starting vLLM with {model}...", flush=True)
    quant_flag = "--quantization fp8" if quantize else ""
    cmd = (
        f"nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate medllm && vllm serve {model} --port 8000 "
        f"--gpu-memory-utilization 0.75 --max-model-len 2048 "
        f"--dtype float16 --enforce-eager {quant_flag}' "
        f"> /tmp/vllm_polish.log 2>&1 &"
    )
    ssh_run(cmd, timeout=5)
    for i in range(30):
        time.sleep(2)
        running, current = check_vllm_server(model)
        if running and current == model:
            print(f"[server] vLLM ready: {model}", flush=True)
            return True
        if i % 5 == 4:
            print(f"[server] Waiting for vLLM... ({(i+1)*2}s)", flush=True)
    print("[server] vLLM failed to start!", flush=True)
    return False


def ensure_servers(need_llm: bool = False, llm_model: str = "Qwen/Qwen3-8B",
                   quantize: bool = True):
    """Ensure all required servers are running. Auto-start/restart as needed."""

    # 1. SSH tunnels
    print("[server] Checking SSH tunnels...", flush=True)
    if not ensure_ssh_tunnel(8787):
        print("[server] Failed to create whisper tunnel (port 8787)")
        return False
    if need_llm and not ensure_ssh_tunnel(8000):
        print("[server] Failed to create LLM tunnel (port 8000)")
        return False

    # 2. Whisper server
    if not check_whisper_server():
        if not start_whisper_server():
            return False
    else:
        print("[server] Whisper server OK", flush=True)

    # 3. vLLM (only if LLM polish enabled)
    if need_llm:
        running, current_model = check_vllm_server(llm_model)
        if running and current_model == llm_model:
            print(f"[server] vLLM OK: {current_model}", flush=True)
        elif running and current_model != llm_model:
            print(f"[server] vLLM has wrong model: {current_model}, need {llm_model}", flush=True)
            stop_vllm_server()
            if not start_vllm_server(llm_model, quantize):
                return False
        else:
            if not start_vllm_server(llm_model, quantize):
                return False

    return True

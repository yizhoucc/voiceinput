from dataclasses import dataclass, field


@dataclass
class Config:
    # STT
    stt_provider: str = "whisper_remote"  # whisper_local | apple_speech | whisper_remote
    whisper_model: str = "small"  # base=fast but bad Chinese, small=good balance, large-v3-turbo needs GPU
    primary_language: str | None = None  # None = auto-detect (restricted to zh/en), "zh" = force Chinese
    stt_step_ms: int = 2000  # sliding window step in ms

    # Whisper initial_prompt: biases decoder toward recognizing these terms as English
    # instead of forcing them into Chinese characters
    whisper_prompt: str = (
        "以下是一段中英文混合的语音转录。"
        "Transformer, attention, QKV, query, key, value, "
        "GPU, CPU, API, Python, JavaScript, TypeScript, "
        "machine learning, deep learning, neural network, "
        "model, token, embedding, encoder, decoder, "
        "GitHub, Docker, Linux, Mac, Windows, "
        "LLM, GPT, BERT, ResNet, CNN, RNN, LSTM, "
        "Whisper, Ollama, vLLM, PyTorch, TensorFlow"
    )

    # Remote whisper (5090 via SSH tunnel)
    whisper_remote_url: str = "http://localhost:8787"

    # LLM Polish
    llm_provider: str = "vllm_remote"  # ollama | vllm_remote
    llm_polish_enabled: bool = True
    ollama_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    vllm_url: str = "http://localhost:8000"
    vllm_model: str = "Qwen/Qwen3-8B"

    # Custom correction dictionary (whisper常见误识别 → 正确词)
    custom_corrections: dict = field(default_factory=lambda: {
        "Cloud": "Claude",
        "探神": "Transformer",
        "详维学习": "强化学习",
        "详话学习": "强化学习",
        "杀铁": "Sharpe",
    })

    # Audio
    sample_rate: int = 16000
    channels: int = 1

    # Hotkey
    hotkey: str = "enter"

    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500  # silence before finalizing a segment


config = Config()

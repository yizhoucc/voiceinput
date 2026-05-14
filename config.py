from dataclasses import dataclass, field


@dataclass
class Config:
    # STT
    stt_provider: str = "whisper_remote"
    whisper_model: str = "small"
    primary_language: str | None = None
    stt_step_ms: int = 2000

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

    whisper_remote_url: str = "http://localhost:8787"

    # LLM
    llm_polish_enabled: bool = False
    vllm_url: str = "http://localhost:8000"
    vllm_model: str = "Qwen/Qwen3-8B"

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

    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500


config = Config()

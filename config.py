from dataclasses import dataclass, field


@dataclass
class Config:
    # STT
    stt_provider: str = "whisper_local"  # whisper_local | apple_speech | whisper_remote
    whisper_model: str = "base"  # use small model for real-time on CPU; large-v3-turbo is too slow without GPU
    primary_language: str | None = None  # None = auto-detect, "zh" = Chinese-primary with English mixed
    stt_step_ms: int = 2000  # sliding window step in ms

    # LLM
    llm_provider: str = "ollama"  # ollama | vllm_remote
    ollama_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    vllm_url: str = "http://10.0.0.145:8000"
    vllm_model: str = "qwen2.5-7b"

    # Audio
    sample_rate: int = 16000
    channels: int = 1

    # Hotkey
    hotkey: str = "option+space"

    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500  # silence before finalizing a segment


config = Config()

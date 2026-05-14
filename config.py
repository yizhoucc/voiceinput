from dataclasses import dataclass, field


@dataclass
class Config:
    # STT
    stt_provider: str = "whisper_remote"  # whisper_local | apple_speech | whisper_remote
    whisper_model: str = "small"  # base=fast but bad Chinese, small=good balance, large-v3-turbo needs GPU
    primary_language: str = "zh"  # "zh" for Chinese-primary with English mixed
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
    hotkey: str = "enter"

    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500  # silence before finalizing a segment


config = Config()

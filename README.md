# VoiceInput

macOS 流式语音输入工具。按快捷键说话，文字实时出现在任意应用的光标位置。

## 功能

- **流式语音输入**：快捷键触发，实时识别，文字插入光标位置
- **中英混合识别**：自动检测中英文，无需手动切换
- **Speaker 识别**：区分"我"和"其他人"，标注 `[我]`/`[他]`（5090 远程模式）
- **LLM 润色**：可选，自动修正谐音错误（穿梳→Transformer、Cloud→Claude）
- **屏幕上下文**：自动 OCR 当前屏幕，提取关键词辅助识别
- **自定义词典**：`dictionary.txt` 配置领域术语和纠正映射
- **批量转录**：将音频/视频文件转为文字文档，支持 txt/srt/tsv 格式

## 安装

```bash
git clone https://github.com/yizhoucc/voiceinput.git
cd voiceinput
python3 -m venv venv
source venv/bin/activate
brew install portaudio
pip install -r requirements.txt
```

### macOS 权限

全局快捷键需要 Accessibility 权限：
**System Settings → Privacy & Security → Accessibility → 添加终端 app**

## 用法

### 实时语音输入

```bash
# 默认：5090 GPU 远程（自动连接，连不上 fallback 到本地 MLX）
python main.py

# 加 LLM 润色
python main.py --llm

# 指定 LLM 模型
python main.py --llm Qwen/Qwen3-8B

# 本地 Mac（MLX Whisper large-v3-turbo，Apple Silicon GPU）
python main.py --local

# 强制语言
python main.py --language zh    # 中文
python main.py --language en    # 英文

# 关闭量化
python main.py --no-quantize
```

**快捷键：**

| 快捷键 | 模式 | 说明 |
|--------|------|------|
| Ctrl+Shift+R | Smart | speaker 切换时自动提交到编辑器 |
| Ctrl+Shift+E | Manual | 停止时才提交（适合长段独白） |

无 Accessibility 权限时：`Enter` = smart，`e`+`Enter` = manual

### 批量转录

```bash
# 单文件（默认带 LLM 润色 + speaker 识别）
python transcribe.py recording.wav

# 视频文件也支持
python transcribe.py video.mp4

# SRT 字幕格式
python transcribe.py recording.wav --format srt

# TSV 表格格式
python transcribe.py recording.wav --format tsv

# 整个目录
python transcribe.py recordings/

# 跳过 LLM 润色
python transcribe.py recording.wav --no-polish

# 本地模式（MLX Whisper，无需 5090）
python transcribe.py recording.wav --local
```

支持格式：wav, mp3, m4a, flac, ogg, aac, mp4, webm

### Speaker 声纹注册

首次使用 speaker 识别前，需要录 15 秒声音注册：

```bash
python enroll_voice.py
```

## 架构

```
默认模式 (5090 远程):
  Mac 麦克风 → sounddevice → 5090 whisper large-v3-turbo (SSH tunnel :8787)
                                      ↓
                                转录 + Speaker 识别
                                      ↓
                                字典纠正 + LLM 润色 (可选, SSH tunnel :8000)
                                      ↓
                                Cmd+V 插入编辑器

本地模式 (--local):
  Mac 麦克风 → sounddevice → MLX Whisper large-v3-turbo (Apple Silicon GPU)
                                      ↓
                                转录 → Cmd+V 插入编辑器
```

### STT 引擎对比

| 引擎 | 准确率 | 速度 (30s音频) | 硬件 |
|------|--------|--------------|------|
| 5090 whisper large-v3-turbo | 100% (基准) | 0.5s | 5090 GPU via SSH |
| MLX Whisper large-v3-turbo | 91% | 5.7s | Mac Apple Silicon GPU |

### 5090 显存占用

| 配置 | 显存 | 命令 |
|------|------|------|
| Whisper only | ~4 GB | `python main.py` |
| Whisper + LLM | ~19 GB | `python main.py --llm` |
| 本地 Mac | 0 (5090) | `python main.py --local` |

### 5090 服务

程序启动时自动管理：
- 检查 SSH 连接 → 连不上自动 fallback 到本地 MLX Whisper
- 检查 whisper server → 没运行自动启动
- 检查 vLLM → 没运行或模型不对自动重启

冷启动约 40 秒（whisper ~20s + vLLM ~20s），之后秒启。

## 自定义词典

编辑 `dictionary.txt`：

```
# 术语（帮助 whisper 识别）
HSTU
Sharpe Ratio
Backpropagation

# 纠正映射（自动替换）
穿梳 -> Transformer
Cloud -> Claude
杀铁 -> Sharpe
```

## 屏幕上下文

录音开始时自动截屏 → macOS Vision OCR → 提取关键词 → 注入 whisper prompt。
屏幕上可见的术语（如 HSTU、Claude）会帮助 whisper 更准确地识别。

## 项目结构

```
main.py              实时语音输入入口
transcribe.py        批量转录 CLI
enroll_voice.py      Speaker 声纹注册
config.py            配置
dictionary.txt       自定义词典
server_manager.py    5090 服务自动管理
audio.py             麦克风采集
audio_utils.py       WAV 转换工具
hotkey.py            全局快捷键
screen_context.py    屏幕 OCR 关键词提取
custom_dict.py       词典加载
stt/
  base.py            STT 抽象接口
  whisper_local.py   MLX Whisper (Mac Apple Silicon GPU)
  whisper_remote.py  5090 GPU faster-whisper via HTTP
llm/
  base.py            LLM 抽象接口
  vllm_remote.py     Qwen3-8B via vLLM
output/
  terminal.py        终端显示 + 日志
  system_insert.py   macOS 文字插入 (Cmd+V)
```

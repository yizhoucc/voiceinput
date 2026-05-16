# VoiceInput

macOS 流式语音输入工具。按快捷键说话，文字实时出现在任意应用的光标位置。

## 功能

- **流式语音输入**：快捷键触发，实时识别，文字插入光标位置
- **中英混合识别**：自动检测中英文，无需手动切换
- **Speaker 识别**：区分"我"和"其他人"，标注 `[我]`/`[他]`
- **LLM 润色**：可选，自动修正谐音错误（穿梳→Transformer、Cloud→Claude）
- **屏幕上下文**：自动 OCR 截屏，提取关键词辅助识别
- **批量转录**：将音频/视频文件转为文字文档，支持 txt/srt/tsv 格式
- **自定义词典**：可配置术语和纠正映射

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
# 默认：5090 GPU 远程识别（自动连接，连不上 fallback 到本地 MLX）
python main.py

# 加 LLM 润色（额外占 ~15GB 显存）
python main.py --llm

# 指定 LLM 模型
python main.py --llm Qwen/Qwen3-8B

# 强制本地 Mac（MLX Whisper on Apple Silicon GPU）
python main.py --local

# 强制语言
python main.py --language zh    # 中文
python main.py --language en    # 英文
```

**快捷键：**

| 快捷键 | 模式 | 说明 |
|--------|------|------|
| Ctrl+Shift+R | Smart | speaker 切换时自动提交到编辑器 |
| Ctrl+Shift+E | Manual | 停止时才提交（适合长段独白） |

无 Accessibility 权限时：`Enter` = smart，`e`+`Enter` = manual

### 批量转录

```bash
# 单文件（自动 LLM 润色 + speaker 识别）
python transcribe.py recording.wav

# SRT 字幕格式（带时间戳）
python transcribe.py recording.wav --format srt

# TSV 表格格式
python transcribe.py recording.wav --format tsv

# 整个目录
python transcribe.py recordings/

# 指定输出路径
python transcribe.py recordings/ -o output.txt

# 跳过 LLM 润色
python transcribe.py recording.wav --no-polish

# 本地模式（MLX Whisper）
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
Mac 麦克风 → sounddevice
                 ↓
         ┌── 5090 模式（默认）──────────────────────┐
         │  SSH tunnel → whisper large-v3-turbo      │
         │             → speaker 识别                 │
         │             → 字典纠正 + LLM 润色 (可选)    │
         └───────────────────────────────────────────┘
         ┌── 本地模式 (--local) ────────────────────┐
         │  MLX Whisper large-v3-turbo (Mac GPU)    │
         │  91% accuracy, 无需 5090                  │
         └───────────────────────────────────────────┘
                 ↓
         Cmd+V 插入编辑器光标
```

### 显存占用（5090 模式）

| 配置 | 5090 显存 | 命令 |
|------|----------|------|
| Whisper only | ~4 GB | `python main.py` |
| Whisper + LLM | ~19 GB | `python main.py --llm` |
| 本地 Mac | 0（用 Mac GPU） | `python main.py --local` |

### 5090 服务自动管理

程序启动时自动管理：
- 检查 SSH 连接 → 连不上自动 fallback 到本地 MLX
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
杀铁 -> Sharpe
Cloud -> Claude
```

## 项目结构

```
main.py              实时语音输入入口
transcribe.py        批量转录 CLI
enroll_voice.py      Speaker 声纹注册
config.py            配置
server_manager.py    5090 服务自动管理
audio.py             麦克风采集
audio_utils.py       WAV 转换工具
hotkey.py            全局快捷键
screen_context.py    截屏 OCR 上下文提取
custom_dict.py       词典加载
dictionary.txt       自定义术语 + 纠正映射
stt/
  base.py            STT 抽象接口
  whisper_local.py   MLX Whisper (Mac Apple Silicon GPU)
  whisper_remote.py  faster-whisper (5090 GPU via SSH)
llm/
  base.py            LLM 抽象接口
  vllm_remote.py     Qwen3-8B via vLLM
output/
  terminal.py        终端显示 + 日志
  system_insert.py   macOS 文字插入
```

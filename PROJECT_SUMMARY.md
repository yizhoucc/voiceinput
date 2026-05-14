# VoiceInput 项目总结

## 项目目标

构建类似 Typeless 的流式语音输入工具，在 macOS 上运行。按快捷键说话，文字实时出现并插入到任意应用的光标位置。

## 架构

```
Mac 麦克风 → 音频 chunks → 5090 GPU (whisper large-v3-turbo)
                                    ↓
                              流式转录 + Speaker 识别
                                    ↓
                              LLM 润色 (Qwen3-8B)
                                    ↓
                              编辑器光标位置插入 (Cmd+V)
```

### 核心设计决策

1. **STT 即时显示 + LLM 异步润色**：whisper 粗转录先显示，LLM 后台修正
2. **全本地方案**：Mac + 5090 LAN，零云端依赖
3. **Append-only 插入**：编辑器文字只追加不修改，避免 backspace/undo 竞态问题
4. **前缀稳定性 commit**：比较连续两次转录的公共前缀，稳定部分自动 commit
5. **Speaker 切换自动 commit**：检测到说话人变化时自动切断并 commit

### 两种录音模式

- **Ctrl+Shift+R (Smart)**：speaker 切换时自动 commit 到编辑器
- **Ctrl+Shift+E (Manual)**：全手动，停止时才 commit

## 技术栈

| 组件 | 技术 | 位置 |
|------|------|------|
| 音频采集 | sounddevice | Mac 本地 |
| STT | faster-whisper large-v3-turbo | 5090 GPU via SSH tunnel |
| Speaker 识别 | speechbrain ECAPA-TDNN | 5090 GPU |
| LLM 润色 | Qwen3-8B via vLLM | 5090 GPU |
| 繁简转换 | opencc | 5090 server |
| 文字插入 | osascript Cmd+V | Mac 本地 |
| 全局快捷键 | pynput | Mac 本地 |

## 性能指标

- STT 延迟：0.5s / 30s 音频（25x 实时）
- LLM 润色：~0.5s / segment
- 全管道端到端：2-4 秒（说完到编辑器出字）
- 内存：O(1)，buffer 定期裁剪

## 质量指标（39 个录音测试）

- Speaker 识别：全部正确
- LLM 修正：7/28 有修正，0 误修
- 关键修正：Cloud→Claude, 探神→Transformer, 详维→强化, 开合→开盒
- GT vs streaming 差距：whisper 滑动窗口固有限制，非管道问题

## 已知限制

1. **streaming 质量 < 全文件质量**：whisper 滑动窗口上下文有限
2. **编辑器插入需要 Accessibility 权限**
3. **5090 必须在线**（SSH 隧道）才能使用远程 STT/LLM
4. **本地 Mac CPU 模式**（small 模型）质量明显低于 5090

## 后续方向

1. **Audio LLM**（Qwen2-Audio）：用 audio tokens + 粗转录一起润色，解决谐音问题
2. **浮动窗口**：Tkinter/SwiftUI overlay 显示实时 partial
3. **Apple Speech 后端**：macOS 原生逐词流式，延迟更低
4. **自动 SSH 隧道管理**：断线重连

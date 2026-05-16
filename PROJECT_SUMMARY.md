# VoiceInput 项目总结

## 项目目标

构建类似 Typeless 的流式语音输入工具，在 macOS 上运行。按快捷键说话，文字实时出现并插入到任意应用的光标位置。

## 架构

```
Mac 麦克风 → sounddevice
                 ↓
         ┌── 5090 模式（默认）──────────────────────┐
         │  SSH tunnel → whisper large-v3-turbo      │
         │             → speaker 识别 (speechbrain)   │
         │             → 字典纠正 + Qwen3-8B 润色     │
         └───────────────────────────────────────────┘
         ┌── 本地模式 (--local) ────────────────────┐
         │  MLX Whisper large-v3-turbo (Mac GPU)    │
         └───────────────────────────────────────────┘
                 ↓
         屏幕 OCR 上下文 → whisper prompt + LLM context
                 ↓
         Cmd+V 插入编辑器光标
```

### 核心设计决策

1. **STT 即时显示 + LLM 异步润色**：whisper 粗转录先显示，LLM 后台修正
2. **全本地方案**：Mac + 5090 LAN，零云端依赖
3. **Append-only 插入**：编辑器文字只追加不修改，避免 backspace/undo 竞态问题
4. **前缀稳定性 commit**：比较连续两次转录的公共前缀，稳定部分自动 commit
5. **Speaker 切换自动 commit**：检测到说话人变化时自动切断并 commit
6. **屏幕上下文**：OCR 截屏提取关键词，注入 whisper prompt 辅助识别

### 两种录音模式

- **Ctrl+Shift+R (Smart)**：speaker 切换时自动 commit 到编辑器
- **Ctrl+Shift+E (Manual)**：停止时才 commit（适合长段独白）

## 技术栈

| 组件 | 技术 | 位置 |
|------|------|------|
| 音频采集 | sounddevice | Mac 本地 |
| STT (远程) | faster-whisper large-v3-turbo | 5090 GPU via SSH tunnel |
| STT (本地) | MLX Whisper large-v3-turbo | Mac Apple Silicon GPU |
| Speaker 识别 | speechbrain ECAPA-TDNN | 5090 GPU |
| LLM 润色 | Qwen3-8B via vLLM | 5090 GPU |
| 繁简转换 | opencc | 5090 server |
| 屏幕上下文 | macOS Vision OCR | Mac 本地 |
| 文字插入 | osascript Cmd+V | Mac 本地 |
| 全局快捷键 | pynput | Mac 本地 |

## 性能指标

### 5090 远程模式
- STT 延迟：0.5s / 30s 音频（25x 实时）
- LLM 润色：~0.5s / segment
- 全管道端到端：2-4 秒
- 准确率：100%（ground truth baseline）

### Mac 本地模式 (MLX Whisper)
- STT 延迟：5.7s / file 平均
- 准确率：91%（字符级重叠）
- 无需网络，无需 5090

### 对比（33 个录音测试）
| 方案 | 字符准确率 | 速度 |
|------|-----------|------|
| 5090 large-v3-turbo | 100% | 0.5s |
| MLX large-v3-turbo (Mac GPU) | 91% | 5.7s |
| faster-whisper small (Mac CPU) | 78% | 4.3s |

## 质量指标

- Speaker 识别：全部正确
- LLM 修正：7/28 有修正，0 误修
- 关键修正：Cloud→Claude, 探神→Transformer, 详维→强化, 开合→开盒
- 量化 (int8) 对质量无影响

## 已知限制

1. **streaming 质量 < 全文件质量**：whisper 滑动窗口上下文有限
2. **编辑器插入需要 Accessibility 权限**
3. **5090 必须在线**才能使用远程 STT/LLM/Speaker 识别
4. **屏幕 OCR 关键词有噪音**：UI 元素和乱码需过滤

## 后续方向

1. 浮动窗口显示 partial（不依赖编辑器文字修改）
2. 等待更成熟的 Audio LLM 用于谐音修正
3. 打包为 .app bundle（可启用 Apple Speech 逐词流式）

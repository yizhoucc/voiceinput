# VoiceInput 开发历程

## 1. 项目定义与架构选型

### 目标
构建类似 Typeless 的 macOS 流式语音输入工具。用户按快捷键说话，文字实时出现在任意应用的光标位置。

### Typeless 机制调研
通过逆向分析 Typeless 的行为特征，排除了三种可能架构：

- **架构 A（串行管道）**：STT 完成 → LLM 完成 → 输出。OpenTypeless 采用此方案。延迟 = STT + LLM，不可接受。
- **架构 B（双流管道）**：STT 流式 + LLM 流式串联。理论可行但两段延迟叠加。
- **架构 C（端到端 Voice LLM）**：Speech ReaLLM 论文的方案，audio embedding 直接喂 LLM。学术前沿，不成熟。

最终确定**架构 B 的变体**：STT partial results 立刻显示，LLM 异步润色后替换。用户先看到粗文字（即时反馈），再看到文字自动变通顺（LLM 追赶）。

### STT 引擎选型
评估了四个方案：

| 方案 | 延迟 | 质量 | 流式类型 |
|------|------|------|---------|
| Deepgram (云端) | <300ms | 高 | WebSocket interim results，真正逐词流式 |
| whisper.cpp (Mac CPU) | 1-3s | 中 | 滑动窗口 chunk 式，每 1-3 秒出一坨 |
| faster-whisper (5090 GPU) | ~0.2s | 高 | 同上但 GPU 极快 |
| Apple Speech (macOS) | <500ms | 中 | 原生逐词流式 |

用户要求全本地、零云端。最终选择 **faster-whisper**：Mac CPU 跑 small 模型做 fallback，5090 GPU 跑 large-v3-turbo 做主力。

**关键发现**：whisper 的"流式"是伪流式——滑动窗口每 N 秒处理一次，不是逐词输出。Deepgram 的 WebSocket interim results 才是真正的逐词流式。但用户优先选择本地方案。

## 2. Phase 1：核心管道搭建

### 2.1 基础骨架
sounddevice 采集麦克风 → faster-whisper 转录 → 终端输出。遇到的第一个问题：

**PyAudio 编译失败**：Meta 内部的 Python 3.12 (fbcode) 链接器找不到 Python 符号。放弃 RealtimeSTT（依赖 PyAudio），改用 sounddevice（ctypes 绑定 portaudio，不需要 C 编译）。

### 2.2 模型选择迭代

| 模型 | 5s 音频处理时间 | 中文质量 | 结论 |
|------|---------------|---------|------|
| large-v3-turbo (CPU) | 4.9s | 好 | **太慢**，无法实时 |
| base (CPU) | 0.31s | 差 | 中文识别错误多 |
| small (CPU) | 0.90s | 中 | **可用**，2s 窗口内 |
| large-v3-turbo (5090 GPU) | 0.04s | 好 | **最优**，25x 实时 |

**关键发现**：faster-whisper 的 CTranslate2 引擎不支持 Apple Silicon GPU（Metal），Mac 上只能 CPU。这决定了 5090 远程方案的必要性。

### 2.3 快捷键困境

尝试了多种全局快捷键方案，每种都有问题：

1. **Option+Space (pynput)**：终端没有 macOS Accessibility 权限，pynput 报 "not trusted"。
2. **NSEvent global monitor (PyObjC)**：同样需要 Accessibility 权限。无法从 Claude Code 终端测试。
3. **双击 Fn 键**：macOS 系统拦截，触发系统听写。
4. **Ctrl+Option+1**：Option 组合键在 macOS 上产生特殊字符（"1" 被打出来）。
5. **F5**：触发 macOS 系统功能（Siri 建议）。
6. **Ctrl+Shift+R**：最终方案。不产生字符，不冲突。

**解决方案**：启动时检测 `AXIsProcessTrusted()`，有权限用 pynput 全局监听，无权限 fallback 到终端 Enter 键。

## 3. 中英文混合识别

### 3.1 问题
用户中英文夹杂说话（"你懂不懂 Transformer，QKV 什么的"），whisper 经常把英文术语错误转成中文谐音："Transformer" → "穿梳"、"传斯福末"。

### 3.2 尝试的方案

**方案 1：`language="zh"` + initial_prompt**
设主语言为中文，用 initial_prompt 包含英文术语列表引导 decoder。效果有限——`base` 模型中文能力太弱，initial_prompt 帮助有限。

**方案 2：双通道 zh+en 合并**
同一段音频跑两次 whisper（一次 `language="zh"`，一次 `language="en"`），按 segment confidence 合并。

结果：en 通道产生大量垃圾文字，confidence 还挺高，混进来污染了中文输出。放弃。

**方案 3：`language=None` 自动检测 + fallback**
让 whisper 自动检测语言，如果检测到非 zh/en（如日文、俄文），用 zh 重跑。

**发现的 bug**：whisper 的 `language=None` 有时检测成俄语（输出西里尔字母 "УФ到底能ины这样"）或日语。根因是自动检测是 per-chunk 的，短 chunk 容易误判。

**最终方案**：`language=None` + server 端 fallback 到 zh + opencc 繁简转换。opencc 是正解（whisper 的 zh 模型不区分简繁），正则过滤是 bandaid（后来删掉了）。

### 3.3 语言翻转问题
滑动窗口重新处理整个 buffer 时，如果用户从中文切到英文，whisper 可能把之前的中文翻译成英文。

**修复**：增量提交机制——文字稳定后"锁定"那段音频，只处理新增音频。防止老内容被重新转录。

## 4. 文字插入：最难的部分

在任意 macOS 应用的光标位置插入和更新文字，是整个项目最困难的部分。经历了 6 种方案，每种都有致命问题。

### 4.1 方案 1：Cmd+Z 撤销 + Cmd+V 粘贴
每次更新：撤销上一次粘贴 → 粘贴新文字。

**问题**：不同应用的 undo 行为不一致。某些应用合并 undo 操作，某些应用的 undo 栈在快速连续调用时混乱。结果：文字丢失。"Transformer" 和 "QKV" 在 terminal 的 final 输出里有，但编辑器里消失了。

### 4.2 方案 2：Shift+Left 选中 + Cmd+V 替换
记录插入的字符数，用 Shift+Left 选中 N 个字符，然后粘贴覆盖。

用 CGEvent 发 Shift+Left：100 个字符 ~72ms，速度可接受。

**问题**：CGEvent 是异步的。发了 N 个 backspace/Shift+Left 事件，但 app 可能只处理了一部分就收到了 paste 事件。`_last_char_count` 和编辑器实际内容不同步，错误累积。

### 4.3 方案 3：增量 diff（公共前缀 + backspace 尾部 + paste 新尾部）
计算新旧文字的公共前缀，只修改变化的尾部。大部分更新只改几个字符。

用 AppleScript（同步执行）替代 CGEvent（异步），消除竞态。

**问题**：即使 AppleScript 同步执行，对于大量 backspace 仍然偶尔失同步。根因：**无法读回编辑器中的实际内容**，所以 `_text_in_editor` 追踪不可靠。一旦失同步，后续所有 diff 都错，错误累积。

### 4.4 方案 4：Cmd+A 全选 + Cmd+V 替换
每次更新全选文本框 → 粘贴整个新文字。原子操作，不需要追踪状态。

**问题**：`Cmd+A` 选中文本框中的所有内容，包括用户之前输入的文字。只适用于空白文本框，不实用。

### 4.5 方案 5：Append-only（只追加，不修改）
放弃所有"修改已插入文字"的方案。编辑器只接收 committed segments（单次 Cmd+V），永不修改。Partial 文字在终端显示。

**效果**：100% 不丢数据。但 commit 间隔太长（8-15 秒），用户体验不好——说了很久编辑器才出字。

### 4.6 最终方案：Append-only + speaker 切换自动 commit + 手动停止 commit
保持 append-only 的可靠性，通过 speaker 切换检测和手动停止触发 commit。

**两种模式**：
- Smart (Ctrl+Shift+R)：speaker 切换自动 commit
- Manual (Ctrl+Shift+E)：只在停止时 commit

## 5. 提交机制的演进

### 5.1 文字稳定性检测
最初方案：连续两次转录结果完全相同 → 提交。

**问题**：连续说话时文字一直在变（whisper 每次重新转录整个 window，segment 边界和文字都变），条件几乎永远不满足。120 秒测试中 0 次中间提交。

### 5.2 stable zone 提交
基于 segment 在 window 中的位置：segment end < window_duration - STABLE_ZONE → 提交。

**问题 1**：STABLE_ZONE 太大（6s）→ 提交太慢。太小（1.5s）→ whisper 上下文不足，"PhD" 被切成 "OPHD"，"强化学习" 被切成 "反强 和学习"。

**问题 2**：buffer trim 切得太多。trim 到 stable_cutoff，但 partial zone 的音频也被裁掉了。finalize 时那段音频已经不存在，内容丢失。

**修复**：只 trim 到最后一个 committed segment 的实际结束时间。

### 5.3 前缀稳定性提交
比较连续两次转录的全文公共前缀。相同的前缀部分是稳定的 → 提交。

**问题**：whisper 每次返回不同粒度的 segments（"嗯" → "嗯,现在好不好" → "嗯,现在好不好用啊,我看看" → …），前缀长度一直在变。当 window 滑动时，window 起点变了，全文完全不同，公共前缀为 0。

**修复**：检测 window 滑动（current_full 不以 committed_text 开头），此时把 previous window 中未提交的内容全部提交。

### 5.4 Force commit 的教训
尝试每 1.5-3 秒强制提交，保证编辑器有输出。

**严重后果**：强制切断 whisper 的上下文窗口。短 chunk 识别质量骤降：
- "PhD" → "OPHD"
- "强化学习" → "反强 和学习"
- "Transformer" → "trafting"
- 大量 "。。" 幻觉

**结论**：不能用 force commit。whisper 需要充足上下文（>5 秒）才能准确识别。

### 5.5 最终方案
取消所有自动提交（停顿、force），只保留 speaker 切换自动提交和手动停止提交。用 ground truth 对比验证。

## 6. Speaker 识别

使用 speechbrain ECAPA-TDNN 声纹编码器。用户录 15 秒声音注册声纹（`enroll_voice.py`），之后每个 segment 计算 cosine similarity，阈值 0.25 区分 me/other。

**效果**：在 roro 双人对话测试中，聚类结果完全正确。cosine similarity 区分度明显：me 0.30-0.61，other -0.18-0.15。

集成到 whisper server，每个 segment 返回 speaker 标签。终端显示 `[我]`/`[他]`，同一 speaker 连续说话时不重复标签。

## 7. LLM 润色

### 7.1 纯文字 LLM 的局限
whisper 把 "Transformer" 转成 "穿梳" 后，纯文字 LLM 看到 "穿梳" 无法推断原词——中文谐音推理对 7B 模型太难。

**验证**：Qwen3-8B 把 "穿梳" 改成了 "穿搭"（合理的中文词），"SWARWY" 改成 "SWAROVSKI"（施华洛世奇）。完全猜错。

**但有 overlap 上下文时效果不同**：给 LLM 同时看到 "穿梳" 和后面的 "QKV, Attention"，LLM 能推断出是 "Transformer"。

### 7.2 自定义纠正词典
在 `config.custom_corrections` 中定义已知的 whisper 误识别映射：
- Cloud → Claude
- 探神 → Transformer
- 详维学习 → 强化学习
- 杀铁 → Sharpe

代码层先做字典替换（instant），然后交给 LLM 做语法和标点修正。

**双重纠正 bug**：最初把词典同时放在代码 `.replace()` 和 LLM system prompt 里，导致 LLM 对已修正的文字再次"修正"。后来删掉了 prompt 里的词典，只在代码层做。

### 7.3 效果验证
28 个有内容的录音文件测试：
- 7 个文件有修正，全部正确
- 0 个误修
- 修正内容：Cloud→Claude (5次), 探神→Transformer, 详维→强化, 开合→开盒, Athropec→Anthropic

### 7.4 Audio LLM 的方向
用户提出的核心洞察：**audio tokens + 粗转录一起喂给 audio LLM**，能解决所有谐音问题。Qwen2-Audio-7B 可以在 5090 上运行（~14GB），与 whisper (~4GB) 共存。这是后续最重要的改进方向。

## 8. 代码审查与清理

三个并行 agent 审查（复用、质量、效率），修复了 13 个文件：

**关键修复**：
- WAV 转换重复 3 处 → 提取 `audio_utils.py`
- `WhisperLocalSTT` 缺 `on_commit` 参数（运行时会崩溃）
- `main.py` 直接操作 `stt._committed_audio_end`（泄漏抽象）→ 加 `prepare_finalize()` 到 base class
- `np.concatenate` 每 100ms 调用一次，O(n²) 复制 → 改为 chunk list
- 4 个回调简化为 2 个 (`on_start(mode)`, `on_stop()`)
- 删除死代码：`_transcribe_range`、`if False:` 块、`overlay.py`、未用方法
- 净减 134 行代码

## 9. 当前系统状态

### 架构
```
Mac 麦克风 → sounddevice → 5090 whisper large-v3-turbo (SSH tunnel)
                                     ↓
                               转录 + speaker 识别
                                     ↓
                               字典纠正 + Qwen3-8B 润色
                                     ↓
                               Cmd+V 插入编辑器光标
```

### 性能
- STT：0.5s / 30s 音频（25x 实时）
- LLM 润色：~0.5s / segment
- 端到端：说完到编辑器出字 2-4 秒

### 已知限制
1. streaming 质量 < 全文件质量（whisper 滑动窗口上下文有限）
2. 编辑器中的文字是 append-only，不做实时更新（所有"修改已插入文字"的方案都不可靠）
3. 纯文字 LLM 无法修正没有上下文线索的谐音错误
4. 5090 必须在线（SSH 隧道），否则退化到 Mac CPU small 模型

### 后续方向
1. Audio LLM（Qwen2-Audio）用 audio tokens 解决谐音问题
2. 浮动窗口显示 partial（不依赖编辑器）
3. Apple Speech 后端（真正逐词流式）
4. SSH 隧道自动管理

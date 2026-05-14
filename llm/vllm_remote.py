import httpx
from llm.base import LLMProvider
from config import config

SYSTEM_PROMPT = """你是语音转录修正助手。修正以下类型的错误：
1. 英文术语被错误转成中文谐音（如"穿梳"应为"Transformer"）
2. 错别字和语音识别错误
3. 标点符号缺失

技术领域：AI/ML（Transformer、Attention、QKV、BERT、GPT、RL、强化学习、Claude、Anthropic等）

规则：
- 只输出修正后的文本
- 保持原意，不添加/删除内容
- 如果有前后文参考（用[上文]标注），利用它推断正确用词
- 保持中英文混合的原始风格
/no_think"""


class VLLMPolisher(LLMProvider):
    def __init__(self):
        self._client = httpx.Client(timeout=30.0)

    def polish(self, text: str, context_before: str = "") -> str:
        if not text.strip():
            return text

        # Dict corrections first (instant, guaranteed)
        corrected = text
        for wrong, right in config.custom_corrections.items():
            corrected = corrected.replace(wrong, right)

        user_msg = ""
        if context_before:
            user_msg += f"[上文] {context_before}\n"
        user_msg += f"[需修正] {corrected}"

        try:
            resp = self._client.post(
                f"{config.vllm_url}/v1/chat/completions",
                json={
                    "model": config.vllm_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": len(corrected) * 2 + 50,
                    "temperature": 0.1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            result = resp.json()
            polished = result["choices"][0]["message"]["content"].strip()
            for tag in ["[需修正]", "[上文]", "[下文]"]:
                polished = polished.replace(tag, "").strip()
            return polished if polished else corrected
        except Exception as e:
            print(f"\n[llm] Polish error: {e}")
            return corrected

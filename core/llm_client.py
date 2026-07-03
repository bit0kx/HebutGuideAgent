from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from anthropic import AsyncAnthropic


@dataclass
class _TextBlock:
    text: str


@dataclass
class _MessageResponse:
    content: List[_TextBlock]


class _OpenAICompatibleMessages:
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **_: Any,
    ) -> _MessageResponse:
        openai_messages: List[Dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        url = f"{self._base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"LLM request failed: {resp.status_code} {resp.text[:500]}")
            data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _MessageResponse(content=[_TextBlock(text=content or "")])


class _OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str):
        self.messages = _OpenAICompatibleMessages(api_key=api_key, base_url=base_url)


def _is_openai_compatible(base_url: Optional[str], model: str) -> bool:
    url = (base_url or "").lower()
    model_name = (model or "").lower()
    return (
        "dashscope.aliyuncs.com" in url
        or "compatible-mode" in url
        or model_name.startswith("qwen")
    )


def create_llm_client(api_key: str, base_url: Optional[str], model: str) -> Any:
    if base_url and _is_openai_compatible(base_url, model):
        return _OpenAICompatibleClient(api_key=api_key, base_url=base_url)

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncAnthropic(**kwargs)

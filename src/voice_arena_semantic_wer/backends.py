from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MODEL = "gemini-3.6-flash"


def parse_verdict(text: str | None) -> bool | None:
    t = (text or "").strip().strip("`*. \n").lower()
    return (t == "true") if t in ("true", "false") else None


class Backend:
    model: str

    def judge(self, prompts: list[str]) -> list[bool | None]:
        raise NotImplementedError


class GeminiSync(Backend):
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL, workers: int = 16, retries: int = 5):
        from google import genai
        from google.genai import types
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Gemini API key missing: pass api_key or set GEMINI_API_KEY")
        self.model, self.workers, self.retries = model, workers, retries
        self._client = genai.Client(api_key=key)
        self._config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="minimal"))

    def _one(self, prompt: str) -> bool | None:
        for attempt in range(self.retries):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=self._config)
                return parse_verdict(resp.text)
            except Exception as e:
                if getattr(e, "code", None) in (400, 401, 403):
                    raise
                time.sleep(2 ** attempt)
        return None

    def judge(self, prompts: list[str]) -> list[bool | None]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, prompts))

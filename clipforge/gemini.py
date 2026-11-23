from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import DEFAULT_MODEL, PROMPTS_DIR


class GeminiError(RuntimeError):
    pass


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 180):
        self.api_key = (api_key or "").strip()
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate_json(self, prompt: str, retries: int = 4) -> dict:
        if not self.available:
            raise GeminiError("GEMINI_API_KEY is not set")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.4,
                },
            }
        ).encode("utf-8")
        last = "unknown error"
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = ""
                for cand in data.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        text += part.get("text", "")
                if not text.strip():
                    raise GeminiError("Gemini returned no text")
                text = text.strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text)
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read()[:400]!r}"
                if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep((2 ** attempt) + random.random())
                    continue
                raise GeminiError(last) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = str(exc)
                if attempt < retries:
                    time.sleep((2 ** attempt) + random.random())
                    continue
                raise GeminiError(last) from exc
        raise GeminiError(last)

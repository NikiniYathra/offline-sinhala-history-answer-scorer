"""Small OLLAMA client for local offline inference."""
from __future__ import annotations

from dataclasses import dataclass
import requests


@dataclass
class OllamaResponse:
    success: bool
    text: str
    error: str | None = None


class OllamaClient:
    def __init__(self, model: str = "Tharusha_Dilhara_Jayadeera/singemma", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, temperature: float = 0.1, timeout: int = 180) -> OllamaResponse:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_ctx": 8192
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return OllamaResponse(success=True, text=data.get("response", "").strip())
        except Exception as exc:
            return OllamaResponse(success=False, text="", error=str(exc))

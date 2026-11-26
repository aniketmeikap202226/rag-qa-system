"""One function, four providers (ollama, gemini, openai, anthropic). Pick with the LLM_PROVIDER environment variable."""
import os
import time

import requests

from src import config


def generate(prompt: str) -> str:
    provider = config.LLM_PROVIDER.lower()

    if provider == "ollama":
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=180,
            )
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama. Install it, run `ollama pull "
                f"{config.OLLAMA_MODEL}`, or set LLM_PROVIDER=openai/anthropic."
            )
        return r.json()["response"].strip()

    if provider == "gemini":
        # Plain REST call, no extra library needed. Key from https://aistudio.google.com
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY first (see README).")
        models = [os.getenv("GEMINI_MODEL", "gemini-3.8-flash")]
        if os.getenv("GEMINI_FALLBACK_MODEL"):
            models.append(os.getenv("GEMINI_FALLBACK_MODEL"))

        last_error = ""
        for model in models:
            for attempt in range(4):  # retry busy/overloaded errors with backoff
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"temperature": 0}},
                    timeout=120,
                )
                if r.status_code == 200:
                    try:
                        parts = r.json()["candidates"][0]["content"]["parts"]
                    except (KeyError, IndexError):
                        raise RuntimeError(f"Gemini returned no answer: {r.text[:300]}")
                    return "".join(p.get("text", "") for p in parts).strip()
                last_error = f"Gemini API error {r.status_code} ({model}): {r.text[:200]}"
                if r.status_code in (429, 500, 503):
                    time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s, 16s
                    continue
                break  # other errors (bad key, wrong model name): do not retry
        raise RuntimeError(
            last_error + "\nTry again later, or set GEMINI_MODEL / GEMINI_FALLBACK_MODEL "
            "to model names from Google AI Studio."
        )

    if provider == "openai":
        from openai import OpenAI  # reads OPENAI_API_KEY
        resp = OpenAI().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

    if provider == "anthropic":
        from anthropic import Anthropic  # reads ANTHROPIC_API_KEY
        resp = Anthropic().messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=600,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

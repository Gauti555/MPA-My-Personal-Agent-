"""
tools/hf_client.py
───────────────────
HuggingFace InferenceClient using provider="auto".

As of 2026, HuggingFace routes large LLM requests through
inference providers (Together AI, Sambanova, Nebius etc).
Every free account gets monthly credits — enough for a weekly blog.

provider="auto" automatically picks the fastest available provider
from your free monthly credits. No paid account needed to start.

Working free models (March 2026):
  - Qwen/Qwen2.5-72B-Instruct        ← best writing quality
  - Qwen/Qwen2.5-7B-Instruct         ← fast, good quality
  - deepseek-ai/DeepSeek-V3-0324     ← excellent reasoning
  - meta-llama/Llama-3.3-70B-Instruct ← great all-rounder
"""

import os
import time
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class HFClient:
    """
    Wrapper around HuggingFace InferenceClient.
    Uses provider="auto" to route through free monthly credits.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        max_new_tokens: int = 1500,
        temperature: float = 0.7,
        timeout: int = 120,
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout

        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise EnvironmentError(
                "HF_TOKEN not found.\n"
                "Set it in your .env file."
            )

        try:
            from huggingface_hub import InferenceClient
            # provider="auto" = uses free monthly credits
            # automatically picks fastest available provider
            self.client = InferenceClient(
                provider="auto",
                api_key=token,
            )
        except ImportError:
            raise ImportError(
                "Run: pip install huggingface-hub==0.32.0"
            )

        logger.info(f"HF client ready — model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=5, max=30),
    )
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generate text. Returns plain string.
        Retries automatically on 503 (model loading) or 429 (rate limit).
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Calling HF ({self.model})...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            text = response.choices[0].message.content or ""
            text = text.strip()
            logger.debug(f"Generated {len(text.split())} words")
            return text

        except Exception as e:
            err = str(e)

            if "503" in err or "loading" in err.lower():
                logger.warning("Model loading — retrying in 15s...")
                time.sleep(15)
                raise

            if "429" in err or "rate" in err.lower():
                logger.warning("Rate limited — retrying in 30s...")
                time.sleep(30)
                raise

            if "402" in err:
                logger.error(
                    f"Model '{self.model}' needs payment.\n"
                    "Try: Qwen/Qwen2.5-7B-Instruct in config.yaml"
                )
                raise

            if "401" in err or "token" in err.lower():
                logger.error(
                    "HF token rejected. Fix:\n"
                    "1. Go to huggingface.co/settings/tokens\n"
                    "2. Edit your token\n"
                    "3. Enable 'Make calls to Inference Providers'\n"
                    "4. Save and update your .env file"
                )
                raise

            logger.error(f"HF error: {e}")
            raise
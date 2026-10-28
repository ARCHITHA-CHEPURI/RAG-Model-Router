import os
import time
from dataclasses import dataclass
from typing import Any

import litellm


class ModelGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_seconds: float


class LiteLLMClient:
    """OpenAI-compatible completion adapter for configured LiteLLM providers."""

    def __init__(
        self,
        model,
        api_base=None,
        api_key=None,
        timeout=180,
        max_tokens=400,
        temperature=0,
        num_ctx=4096,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model must be a non-empty string.")
        if timeout <= 0 or max_tokens < 1 or num_ctx < 1:
            raise ValueError("Timeout, max_tokens, and num_ctx must be positive.")

        self.model = model.strip()
        self.display_name = (
            self.model.removeprefix("ollama_chat/")
            .removeprefix("ollama/")
        )
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_ctx = num_ctx

    def generate_response(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.model.startswith(("ollama_chat/", "ollama/")):
            kwargs["num_ctx"] = self.num_ctx

        start_time = time.perf_counter()
        try:
            response = litellm.completion(**kwargs)
        except (
            litellm.APIError,
            litellm.APIConnectionError,
            litellm.Timeout,
        ) as error:
            raise ModelGenerationError(
                f"LiteLLM request failed for {self.display_name}: {error}"
            ) from error
        latency = time.perf_counter() - start_time

        try:
            choice = response.choices[0]
            content = choice.message.content
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ModelGenerationError(
                f"LiteLLM returned an invalid response for {self.display_name}."
            ) from error

        if not isinstance(content, str) or not content.strip():
            raise ModelGenerationError(
                f"LiteLLM returned an empty response for {self.display_name}."
            )

        usage = getattr(response, "usage", None)
        input_tokens = self._read_usage(usage, "prompt_tokens")
        output_tokens = self._read_usage(usage, "completion_tokens")
        cost_usd = (
            None
            if self.model.startswith(("ollama_chat/", "ollama/"))
            else self._read_cost(response)
        )

        return ModelResponse(
            text=content.strip(),
            model=self.display_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_seconds=latency,
        )

    def generate(self, prompt):
        """Keep the original text-only interface for CLI and benchmark callers."""
        return self.generate_response(prompt).text

    @staticmethod
    def _read_usage(usage, key):
        if usage is None:
            return None
        value = (
            usage.get(key)
            if isinstance(usage, dict)
            else getattr(usage, key, None)
        )
        return value if isinstance(value, int) and value >= 0 else None

    @staticmethod
    def _read_cost(response):
        hidden_params = getattr(response, "_hidden_params", None)
        if not isinstance(hidden_params, dict):
            return None
        value = hidden_params.get("response_cost")
        return value if isinstance(value, (int, float)) and value >= 0 else None


class OllamaClient(LiteLLMClient):
    """Backward-compatible Ollama adapter, now implemented through LiteLLM."""

    def __init__(
        self,
        model="llama3.2:1b",
        base_url=None,
        **kwargs,
    ):
        resolved_base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        provider_model = (
            model
            if model.startswith(("ollama_chat/", "ollama/"))
            else f"ollama_chat/{model}"
        )
        super().__init__(
            model=provider_model,
            api_base=resolved_base_url.removesuffix("/v1"),
            **kwargs,
        )

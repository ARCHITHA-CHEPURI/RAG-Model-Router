import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import litellm

from llm.llm_router import LLMRouter
from llm.ollama_client import (
    LiteLLMClient,
    ModelGenerationError,
    OllamaClient,
)


def completion_response(
    text="Hello",
    prompt_tokens=7,
    completion_tokens=2,
    cost=0.0003,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        _hidden_params={"response_cost": cost},
    )


class LiteLLMAdapterTests(unittest.TestCase):
    @patch("llm.ollama_client.litellm.completion")
    def test_ollama_adapter_uses_litellm_chat_contract(self, completion):
        completion.return_value = completion_response()
        client = OllamaClient(
            model="qwen2.5:0.5b",
            base_url="http://localhost:11434",
        )

        result = client.generate_response("Say hello")

        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.model, "qwen2.5:0.5b")
        self.assertEqual(result.input_tokens, 7)
        self.assertEqual(result.output_tokens, 2)
        self.assertIsNone(result.cost_usd)
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "ollama_chat/qwen2.5:0.5b")
        self.assertEqual(kwargs["api_base"], "http://localhost:11434")
        self.assertEqual(
            kwargs["messages"],
            [{"role": "user", "content": "Say hello"}],
        )
        self.assertEqual(kwargs["max_tokens"], 400)
        self.assertEqual(kwargs["num_ctx"], 4096)

    @patch("llm.ollama_client.litellm.completion")
    def test_text_generate_preserves_legacy_client_contract(self, completion):
        completion.return_value = completion_response(text="  Hello  ")
        client = OllamaClient()

        self.assertEqual(client.generate("Say hello"), "Hello")

    @patch("llm.ollama_client.litellm.completion")
    def test_generic_cloud_adapter_uses_supplied_provider_config(self, completion):
        completion.return_value = completion_response()
        client = LiteLLMClient(
            model="openai/gpt-4o-mini",
            api_base="https://provider.example/v1",
            api_key="test-key",
        )

        result = client.generate_response("Say hello")

        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
        self.assertEqual(kwargs["api_base"], "https://provider.example/v1")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertNotIn("num_ctx", kwargs)
        self.assertEqual(result.cost_usd, 0.0003)

    @patch("llm.ollama_client.litellm.completion")
    def test_adapter_surfaces_provider_errors(self, completion):
        completion.side_effect = litellm.APIConnectionError(
            message="offline",
            llm_provider="ollama",
            model="qwen2.5:0.5b",
        )
        client = OllamaClient()

        with self.assertRaisesRegex(ModelGenerationError, "LiteLLM request failed"):
            client.generate("Say hello")

    @patch("llm.ollama_client.litellm.completion")
    def test_adapter_rejects_empty_model_output(self, completion):
        completion.return_value = completion_response(text=" ")

        with self.assertRaisesRegex(ModelGenerationError, "empty response"):
            LiteLLMClient(model="openai/gpt-4o-mini").generate("Say hello")

    def test_router_reads_provider_configuration_from_environment(self):
        environment = {
            "ROUTER_SMALL_MODEL": "openai/gpt-4o-mini",
            "ROUTER_SMALL_API_KEY": "test-key",
            "ROUTER_SMALL_API_BASE": "https://provider.example/v1",
            "ROUTER_LARGE_MODEL": "ollama_chat/llama3.2:1b",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
        with patch.dict(os.environ, environment, clear=True):
            router = LLMRouter()

        self.assertEqual(router.model_names["small"], "openai/gpt-4o-mini")
        self.assertEqual(router.clients["small"].api_key, "test-key")
        self.assertEqual(
            router.clients["small"].api_base,
            "https://provider.example/v1",
        )
        self.assertEqual(router.model_names["large"], "llama3.2:1b")
        self.assertIsNone(router.clients["large"].api_key)
        self.assertEqual(
            router.clients["large"].api_base,
            "http://localhost:11434",
        )

    def test_adapter_rejects_blank_prompts(self):
        with self.assertRaises(ValueError):
            LiteLLMClient(model="openai/gpt-4o-mini").generate(" ")


if __name__ == "__main__":
    unittest.main()

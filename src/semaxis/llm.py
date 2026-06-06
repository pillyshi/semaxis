from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar, runtime_checkable

import tiktoken
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class BaseLLMClient(Protocol):
    """Protocol defining the interface required by all LLM client implementations."""

    def complete(self, messages: list[dict[str, str]]) -> str: ...
    def complete_json(self, messages: list[dict[str, str]]) -> Any: ...
    def complete_structured(self, messages: list[dict[str, str]], response_model: type[T]) -> T: ...
    def count_tokens(self, text: str) -> int: ...


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API with token counting."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request and return the content string."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""

    def complete_json(self, messages: list[dict[str, str]]) -> Any:
        """Send a request expecting JSON output and return the parsed object."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},  # type: ignore[call-overload]
        )
        content = response.choices[0].message.content or ""
        return json.loads(content)

    def complete_structured(self, messages: list[dict[str, str]], response_model: type[T]) -> T:
        """Send a structured output request and return a validated Pydantic model instance."""
        response = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            response_format=response_model,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned a refusal or null parsed response for structured output")
        return parsed  # type: ignore[return-value]

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text using the model's tokenizer."""
        return len(self._encoding.encode(text))


class LlamaCppClient:
    """LLM client backed by a llama-cpp-python Llama instance for in-process inference.

    Example::

        from llama_cpp import Llama
        client = LlamaCppClient(Llama(model_path="path/to/model.gguf", n_ctx=4096))
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        result = self._model.create_chat_completion(messages=messages)
        return result["choices"][0]["message"]["content"] or ""

    def complete_json(self, messages: list[dict[str, str]]) -> Any:
        result = self._model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = result["choices"][0]["message"]["content"] or ""
        # Use _extract_json rather than json.loads directly: grammar enforcement quality
        # varies by model and quantization, so some outputs may include markdown fences.
        return _extract_json(content)

    def complete_structured(self, messages: list[dict[str, str]], response_model: type[T]) -> T:
        schema = response_model.model_json_schema()
        result = self._model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
        )
        content = result["choices"][0]["message"]["content"] or ""
        data = _extract_json(content)
        return response_model.model_validate(data)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenize(text.encode()))


def _extract_json(text: str) -> Any:
    """Extract a JSON object from a string, tolerating surrounding prose.

    Tries in order:
    1. Direct json.loads (model output is clean JSON)
    2. Extract from a markdown code block (```json ... ```)
    3. Extract the first {...} or [...] block via regex
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    code_block = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Try first {...} or [...] block
    brace_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from model response:\n{text}")

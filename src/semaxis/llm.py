from __future__ import annotations

import json
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
        return _chat_content(result)

    def complete_json(self, messages: list[dict[str, str]]) -> Any:
        result = self._model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
        )
        return json.loads(_chat_content(result))

    def complete_structured(self, messages: list[dict[str, str]], response_model: type[T]) -> T:
        # Inline $ref pointers so llama-cpp's grammar converter sees a fully resolved schema.
        schema = _inline_refs(response_model.model_json_schema())
        result = self._model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
        )
        return response_model.model_validate(json.loads(_chat_content(result)))

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenize(text.encode(), add_bos=False))


def _chat_content(result: dict[str, Any]) -> str:
    return result["choices"][0]["message"]["content"] or ""


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve $ref pointers against $defs, returning a flat schema.

    llama-cpp-python's grammar converter may not resolve $ref in all versions,
    so we inline before passing the schema.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref: str = node["$ref"]
                if ref.startswith("#/$defs/"):
                    return resolve(defs[ref[len("#/$defs/"):]])
                return node
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)  # type: ignore[return-value]

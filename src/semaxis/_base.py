from __future__ import annotations

from typing import Self


class _LLMTransformerMixin:
    """Mixin that makes sklearn clone() work with non-picklable LLM client instances."""

    def __sklearn_clone__(self) -> Self:
        return type(self)(**self.get_params(deep=False))  # type: ignore[attr-defined]

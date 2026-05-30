from __future__ import annotations

import copy
from typing import Self


class _LLMTransformerMixin:
    """Mixin for sklearn clone() compatibility with non-picklable LLM client instances.

    Must be combined with ``sklearn.base.BaseEstimator`` (which provides ``get_params``).
    """

    def __sklearn_clone__(self) -> Self:
        new = type(self)(**self.get_params(deep=False))  # type: ignore[attr-defined]
        # Preserve set_output() and metadata routing config, mirroring what
        # BaseEstimator._clone_parametrized copies after construction.
        if hasattr(self, "_sklearn_output_config"):
            new._sklearn_output_config = copy.deepcopy(self._sklearn_output_config)
        if hasattr(self, "_metadata_request"):
            new._metadata_request = copy.deepcopy(self._metadata_request)
        return new

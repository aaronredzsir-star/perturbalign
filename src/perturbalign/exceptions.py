"""PerturbAlign unified domain exception hierarchy.

Provides structured, self-explanatory domain exceptions instead of bare ValueError/KeyError,
enabling callers and AI Agents to handle, pinpoint, and recover from domain errors deterministically.
"""
from __future__ import annotations

import re
from typing import Any, Optional


class PerturbAlignError(Exception):
    """Top-level base class for all PerturbAlign domain exceptions."""
    def __init__(
        self,
        message: str = "",
        details: Optional[dict[str, Any]] = None,
        entity_name: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.entity_name = entity_name
        self.error_code = error_code or self._default_error_code()
        if entity_name:
            self.details["entity_name"] = entity_name
        if self.error_code:
            self.details["error_code"] = self.error_code

    def _default_error_code(self) -> str:
        s = re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).upper()
        return s

    def __str__(self) -> str:
        base = self.message
        if self.entity_name and self.entity_name not in base:
            return f"{base} (entity: '{self.entity_name}')"
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "entity_name": self.entity_name,
            "details": self.details,
        }


class NormalizationError(PerturbAlignError):
    """Exception raised during perturbation entity cleaning, alignment, or disambiguation."""
    pass


class AmbiguousEntityError(NormalizationError):
    """Raised when an entity name matches multiple high-confidence candidates without clear resolution."""
    def __init__(
        self,
        query: str = "",
        candidates: Optional[list[Any]] = None,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        candidates = candidates or []
        msg = message or f"Perturbation label '{query}' is ambiguous with {len(candidates)} candidate entities; no silent guessing."
        entity_name = kwargs.pop("entity_name", query)
        super().__init__(
            message=msg,
            entity_name=entity_name,
            details={"query": query, "candidate_count": len(candidates)},
            **kwargs,
        )
        self.query = query
        self.candidates = candidates


class UnmatchedEntityError(NormalizationError):
    """Raised when an entity cannot be matched in current mode (e.g. offline-only)."""
    def __init__(
        self,
        query: str = "",
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        msg = message or f"Failed to resolve perturbation entity in current corpus or external knowledge bases: '{query}'"
        entity_name = kwargs.pop("entity_name", query)
        super().__init__(
            message=msg,
            entity_name=entity_name,
            details={"query": query},
            **kwargs,
        )
        self.query = query


class SpatiotemporalParseError(PerturbAlignError):
    """Raised when parsing dose, duration, or physical units fails."""
    def __init__(
        self,
        raw_input: str = "",
        reason: str = "",
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        msg = message or (f"Failed to parse spatiotemporal dose/duration: '{raw_input}'" + (f" ({reason})" if reason else ""))
        entity_name = kwargs.pop("entity_name", raw_input)
        super().__init__(
            message=msg,
            entity_name=entity_name,
            details={"raw_input": raw_input, "reason": reason},
            **kwargs,
        )
        self.raw_input = raw_input
        self.reason = reason


class DimensionalMismatchError(PerturbAlignError):
    """Raised when physical dimensions are incompatible (e.g. molar to mass without molecular weight)."""
    def __init__(
        self,
        source_unit: str = "",
        target_unit: str = "",
        reason: str = "",
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        msg = message or (f"Cannot convert between dimensions '{source_unit}' and '{target_unit}'" + (f": {reason}" if reason else ""))
        super().__init__(
            message=msg,
            details={"source_unit": source_unit, "target_unit": target_unit, "reason": reason},
            **kwargs,
        )
        self.source_unit = source_unit
        self.target_unit = target_unit
        self.reason = reason


class RegistryError(PerturbAlignError):
    """Raised when a KG provider or protocol adapter registration encounters conflict or missing name."""
    def __init__(
        self,
        name: str = "",
        reason: str = "",
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        msg = message or f"Registry exception for '{name}': {reason}"
        entity_name = kwargs.pop("entity_name", name)
        super().__init__(
            message=msg,
            entity_name=entity_name,
            details={"provider_name": name, "reason": reason},
            **kwargs,
        )
        self.name = name
        self.reason = reason


class GraphInferenceError(PerturbAlignError):
    """Raised on heterogeneous knowledge graph multi-hop causal inference or traversal failure."""
    def __init__(
        self,
        source: str = "",
        target: str = "",
        reason: str = "",
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        msg = message or f"Causal inference in knowledge graph from '{source}' to '{target}' failed: {reason}"
        super().__init__(
            message=msg,
            details={"source": source, "target": target, "reason": reason},
            **kwargs,
        )
        self.source = source
        self.target = target
        self.reason = reason
